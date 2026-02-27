"""
DreamShift dLLM decoding algorithm.

Implements block-by-block denoising with scheduled token transfer,
matching the `block_diffusion_generate_with_shift` + `low_confidence_dynamic`
logic from the reference training/generation code.

Key differences vs the plain `LowConfidence` algorithm:
  - Fixed `denoising_steps` per block (not adaptive up-to-block_size).
  - Token transfer per step is *scheduled* by `get_num_transfer_tokens`
    (distributes block_size tokens as evenly as possible across steps),
    rather than always transferring every high-confidence token.
  - `low_confidence_dynamic`:
      if |high_conf_mask| >= n_transfer  →  unmask ALL high-conf tokens
      else                               →  unmask top-n_transfer by prob
  - Logit shift (when config.causal_prefill is True):
      DreamShift models are trained with hidden[i] predicting token[i+1].
      At inference, to predict token at position i, we use logit[i-1].
      Shift: shifted_logits = cat([prev_block_last_logits, raw_logits[:-1]])
      prev_block_last_logits is saved after each STAGING_PREFILL / commit pass.

Config keys (passed via --dllm-algorithm-config YAML):
  denoising_steps:       int,   default 8
  threshold:             float, default 0.85
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


def get_num_transfer_tokens(block_size: int, steps: int) -> torch.Tensor:
    """
    Distribute `block_size` tokens evenly across `steps` denoising steps.
    Remainder tokens are added to the first steps.

    Example: block_size=8, steps=3  →  [3, 3, 2]
    """
    base = block_size // steps
    remainder = block_size % steps
    schedule = torch.full((steps,), base, dtype=torch.int64)
    schedule[:remainder] += 1
    return schedule


class DreamShift(DllmAlgorithm):
    """
    Block-diffusion decoding with scheduled low-confidence-dynamic unmasking.

    The algorithm processes one block at a time (as supplied by the dLLM
    scheduler). For each block it runs exactly `denoising_steps` forward
    passes (with early exit when no mask tokens remain) and transfers
    tokens following the `low_confidence_dynamic` rule:

      1. Sample x0 = argmax(shifted_logits) for each masked position.
      2. Compute per-token probability p.
      3. high_conf_mask = (p > threshold) & mask_index
      4. If |high_conf_mask| >= n_transfer:  transfer = high_conf_mask
         Else:                               transfer = top-n_transfer(p)

    For DreamShift-trained models (config.causal_prefill=True), applies
    the logit shift: shifted[i] = raw[i-1], shifted[0] = prev_block_last.
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.denoising_steps: int = config.algorithm_config.get("denoising_steps", 8)
        self.threshold: float = config.algorithm_config.get("threshold", 0.85)
        # Pre-compute the per-step token-transfer schedule
        self._schedule = get_num_transfer_tokens(self.block_size, self.denoising_steps)
        # Whether to apply the Dream-style logit shift at inference.
        # Required for models trained with hidden[i] predicting token[i+1]
        # (i.e. DreamShift / SDAR models with use_regular_causal=True).
        self.apply_logit_shift: bool = config.causal_prefill
        # Per-request storage for the last logit from the previous block.
        # Key: req_pool_index (int), Value: logit tensor of shape [vocab_size].
        # Populated after STAGING_PREFILL and each block's commit pass.
        self._prev_last_logits: Dict[int, torch.Tensor] = {}

    def _get_shifted_logits(
        self,
        full_logits: torch.Tensor,
        batch_id: int,
        req_pool_idx: int,
    ) -> torch.Tensor:
        """
        Return shifted logits for one request's block.

        For DreamShift models: shifted[i] = raw[i-1], shifted[0] = prev_last.
        Falls back to raw[0] as prev_last if no prior state is available.

        To avoid batch-position-dependent numerical divergence in the paged
        attention commit pass, we reconstruct a sharp pseudo-logit from the
        argmax of _prev_last_logits instead of using the raw logit vector.
        This makes shifted[0]'s confidence deterministic (~1.0) while
        preserving the correct token prediction.
        """
        raw_block = full_logits[
            batch_id * self.block_size : (batch_id + 1) * self.block_size
        ]  # [block_size, vocab]
        prev_last = self._prev_last_logits.get(req_pool_idx, None)
        if prev_last is None:
            # Fallback: no STAGING_PREFILL ran (very short prompt).
            # Mirror the reference: use the first logit as the "previous".
            prev_last = raw_block[0]
        else:
            # Reconstruct a sharp pseudo-logit from the saved argmax.
            # The commit pass logit can differ across batch positions due to
            # paged attention FP16 precision, which causes the softmax
            # confidence at shifted[0] to cross the threshold differently
            # across batch positions. Using a sharp pseudo-logit makes the
            # confidence deterministic (~1.0), eliminating this divergence
            # while preserving the correct argmax prediction.
            argmax_tok = prev_last.argmax()
            prev_last = torch.full_like(prev_last, -100.0)
            prev_last[argmax_tok] = 100.0
        shifted = torch.cat([prev_last.unsqueeze(0), raw_block[:-1]], dim=0)
        return shifted  # [block_size, vocab]

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device

        # Move schedule to device on first use
        schedule = self._schedule.to(device)

        # Track the start of generated tokens for each batch item
        start_list = []
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            n_masked = int((block_input_ids == self.mask_id).sum().item())
            start_list.append(self.block_size - n_masked)

        # Fast path: no mask tokens in the entire batch → STAGING_PREFILL
        mask_index_global = forward_batch.input_ids == self.mask_id
        if not mask_index_global.any():
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            if self.apply_logit_shift:
                # Save the last logit of each request's prefill sequence.
                # For DreamShift: logit[-1] predicts the first token of the
                # next (decode) block.
                #
                # full_logits has shape [total_new_tokens, vocab]. In
                # STAGING_PREFILL each request contributes (seq_len -
                # prefix_len) NEW tokens; seq_lens counts the total including
                # cached prefix, so we compute n_new per request.
                full_logits = out.logits_output.full_logits
                prefix_lens = forward_batch.extend_prefix_lens  # may be None
                offset = 0
                pool_idxs_debug = []
                for bid in range(batch_size):
                    seq_len = int(forward_batch.seq_lens[bid].item())
                    prefix_len = (
                        int(prefix_lens[bid].item())
                        if prefix_lens is not None
                        else 0
                    )
                    n_new = seq_len - prefix_len
                    req_pool_idx = int(forward_batch.req_pool_indices[bid].item())
                    last_idx = offset + n_new - 1
                    self._prev_last_logits[req_pool_idx] = (
                        full_logits[last_idx].detach().clone()
                    )
                    pool_idxs_debug.append(req_pool_idx)
                    offset += n_new
                if batch_size <= 16:
                    top3 = [int(self._prev_last_logits[p].argmax().item()) for p in pool_idxs_debug]
                    import sys; print(f"[DS PREFILL] bs={batch_size} full_logits={full_logits.shape[0]} pool={pool_idxs_debug} top_tok={top3}", flush=True)
            return out.logits_output, [], out.can_run_graph

        # Denoising loop: exactly denoising_steps steps (with early exit)
        logits_output = None
        can_run_cuda_graph = True

        if batch_size <= 16:
            n_masks_per = []
            for bid in range(batch_size):
                bs2 = bid * self.block_size
                be2 = bs2 + self.block_size
                n_masks_per.append(int((forward_batch.input_ids[bs2:be2] == self.mask_id).sum().item()))
            pool_debug = [int(forward_batch.req_pool_indices[i].item()) for i in range(batch_size)]
            prev_debug = {p: int(self._prev_last_logits[p].argmax().item()) if p in self._prev_last_logits else -1 for p in pool_debug}
            import sys; print(f"[DS DECODE] bs={batch_size} masks={n_masks_per} pool={pool_debug} start={start_list} prev_top={prev_debug}", flush=True)

        for step in range(self.denoising_steps):
            mask_index_global = forward_batch.input_ids == self.mask_id
            if not mask_index_global.any():
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            raw_full_logits = logits_output.full_logits  # [batch*block_size, vocab]

            n_transfer = int(schedule[step].item())

            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size

                block_input_ids = forward_batch.input_ids[curr_block_start:curr_block_end]
                block_mask_index = block_input_ids == self.mask_id

                if not block_mask_index.any():
                    continue

                req_pool_idx = int(forward_batch.req_pool_indices[batch_id].item())

                if self.apply_logit_shift:
                    curr_logits = self._get_shifted_logits(
                        raw_full_logits, batch_id, req_pool_idx
                    )
                else:
                    curr_logits = raw_full_logits[curr_block_start:curr_block_end]

                # Greedy prediction and probability for each position
                x0 = torch.argmax(curr_logits, dim=-1)
                p = F.softmax(curr_logits, dim=-1).gather(
                    -1, x0.unsqueeze(-1)
                ).squeeze(-1)

                # low_confidence_dynamic selection
                confidence = torch.where(
                    block_mask_index, p, torch.tensor(-np.inf, device=device)
                )
                high_conf_mask = confidence > self.threshold
                n_high = int(high_conf_mask.sum().item())

                num_to_transfer = min(n_transfer, int(block_mask_index.sum().item()))
                if num_to_transfer == 0:
                    continue

                if n_high >= num_to_transfer:
                    transfer_index = high_conf_mask
                else:
                    transfer_index = torch.zeros_like(block_mask_index)
                    _, idx = torch.topk(confidence, num_to_transfer)
                    transfer_index[idx] = True

                block_input_ids[transfer_index] = x0[transfer_index]

        # Final forward pass to commit KV cache for the completed block
        if batch_size <= 16:
            seq_lens_dbg = forward_batch.seq_lens[:batch_size].tolist()
            pref_dbg = forward_batch.extend_prefix_lens[:batch_size].tolist() if forward_batch.extend_prefix_lens is not None else "None"
            ids_per_req = [forward_batch.input_ids[i*self.block_size:(i+1)*self.block_size].tolist() for i in range(batch_size)]
            import sys; print(f"[DS PRE-COMMIT] bs={batch_size} seq_lens={seq_lens_dbg} prefix_lens={pref_dbg} ids={ids_per_req}", flush=True)
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        if self.apply_logit_shift:
            # Save the last RAW logit of each request's block for the next
            # block's shift.  Use commit-pass logits (not shifted) so that
            # logit[-1] correctly predicts the first token of the next block.
            #
            # IMPORTANT: Only update _prev_last_logits for requests that had
            # masks in their block (STAGING_DECODE). For requests with 0 masks
            # (STAGING_PREFILL), the correct causal logit was already saved
            # during the fast path. Overwriting it here with a logit from the
            # DLLM_EXTEND (bidirectional) forward pass would corrupt the next
            # block's shift.  start_list[i] = block_size - n_masked_at_entry,
            # so start_list[i] < block_size  ⟺  block had masks.
            commit_full_logits = logits_output.full_logits
            for batch_id in range(batch_size):
                if start_list[batch_id] >= self.block_size:
                    # No masks in this request's block — skip to preserve the
                    # causal _prev_last_logits set by the STAGING_PREFILL path.
                    continue
                req_pool_idx = int(forward_batch.req_pool_indices[batch_id].item())
                block_last_idx = (batch_id + 1) * self.block_size - 1
                self._prev_last_logits[req_pool_idx] = (
                    commit_full_logits[block_last_idx].detach().clone()
                )
            if batch_size <= 16:
                committed = [(int(forward_batch.req_pool_indices[i].item()), start_list[i]) for i in range(batch_size) if start_list[i] < self.block_size]
                # Check if logits differ across batch positions
                top_toks = []
                for bid in range(batch_size):
                    if start_list[bid] < self.block_size:
                        bli = (bid + 1) * self.block_size - 1
                        top_toks.append(int(commit_full_logits[bli].argmax().item()))
                # Check if logits at different batch positions are identical
                if batch_size >= 2 and all(s < self.block_size for s in start_list):
                    idx0 = self.block_size - 1
                    idx1 = 2 * self.block_size - 1
                    diff = (commit_full_logits[idx0] - commit_full_logits[idx1]).abs().max().item()
                    import sys; print(f"[DS COMMIT] bs={batch_size} top_toks={top_toks} logit_diff_01={diff:.6f}", flush=True)
                else:
                    import sys; print(f"[DS COMMIT] bs={batch_size} committed={committed} top_toks={top_toks}", flush=True)

        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i]:] for i in range(batch_size)
        ]

        if batch_size <= 16:
            for bid in range(batch_size):
                pidx = int(forward_batch.req_pool_indices[bid].item())
                toks = next_token_ids_list[bid].tolist()
                import sys; print(f"[DS TOKENS] pool={pidx} start={start_list[bid]} toks={toks}", flush=True)

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = DreamShift
