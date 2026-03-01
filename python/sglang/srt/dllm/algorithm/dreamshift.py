"""
DreamShift dLLM decoding algorithm.

Implements block-by-block denoising with scheduled token transfer,
matching the `block_diffusion_generate_with_shift` + `low_confidence_dynamic`
logic from the reference training/generation code.

Config keys (passed via --dllm-algorithm-config YAML):
  denoising_steps:       int,   default 8
  threshold:             float, default 0.85
  temperature:           float, default 1.0  (0 = greedy/argmax)
  top_k:                 int,   default 0    (0 = disabled)
  top_p:                 float, default 1.0  (1.0 = disabled)
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


def get_num_transfer_tokens(block_size: int, steps: int) -> torch.Tensor:
    """Distribute `block_size` tokens evenly across `steps` denoising steps."""
    base = block_size // steps
    remainder = block_size % steps
    schedule = torch.full((steps,), base, dtype=torch.int64)
    schedule[:remainder] += 1
    return schedule


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out logits below the top-k threshold."""
    if k <= 0:
        return logits
    top_k_val = logits.topk(k, dim=-1).values[:, -1:]
    return logits.masked_fill(logits < top_k_val, -float("inf"))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out logits below the nucleus (top-p) threshold."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    cumsum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    mask = cumsum - sorted_logits.softmax(dim=-1) >= p
    sorted_logits[mask] = -float("inf")
    return sorted_logits.scatter(-1, sorted_idx, sorted_logits)


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample tokens from logits, matching reference generate.py's
    `sample_with_temperature_topk_topp`.

    Returns:
        x0: sampled token ids [block_size]
        x0_p: probability of sampled tokens [block_size]
    """
    if temperature <= 0:
        # Greedy / argmax
        probs = F.softmax(logits, dim=-1)
        x0 = probs.argmax(dim=-1)
        x0_p = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        return x0, x0_p

    filtered = logits
    if top_k > 0:
        filtered = _top_k_filter(filtered, top_k)
    if top_p < 1.0:
        filtered = _top_p_filter(filtered, top_p)
    if temperature != 1.0:
        filtered = filtered / temperature

    probs = F.softmax(filtered, dim=-1)
    x0 = torch.multinomial(probs, num_samples=1).squeeze(-1)
    x0_p = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    return x0, x0_p


class DreamShift(DllmAlgorithm):
    """
    Block-diffusion decoding with scheduled low-confidence-dynamic unmasking.
    Supports temperature/top_k/top_p sampling to match reference implementation.
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.denoising_steps: int = config.algorithm_config.get("denoising_steps", 8)
        self.threshold: float = config.algorithm_config.get("threshold", 0.85)
        self.temperature: float = config.algorithm_config.get("temperature", 1.0)
        self.top_k: int = config.algorithm_config.get("top_k", 0)
        self.top_p: float = config.algorithm_config.get("top_p", 1.0)
        self._schedule = get_num_transfer_tokens(self.block_size, self.denoising_steps)
        self.apply_logit_shift: bool = config.causal_prefill
        self._prev_last_logits: Dict[int, torch.Tensor] = {}

    def _get_shifted_logits(
        self,
        full_logits: torch.Tensor,
        batch_id: int,
        req_pool_idx: int,
    ) -> torch.Tensor:
        """Return shifted logits: shifted[i] = raw[i-1], shifted[0] = prev_last."""
        raw_block = full_logits[
            batch_id * self.block_size : (batch_id + 1) * self.block_size
        ]
        prev_last = self._prev_last_logits.get(req_pool_idx, None)
        if prev_last is None:
            prev_last = raw_block[0]
        else:
            argmax_tok = prev_last.argmax()
            prev_last = torch.full_like(prev_last, -100.0)
            prev_last[argmax_tok] = 100.0
        shifted = torch.cat([prev_last.unsqueeze(0), raw_block[:-1]], dim=0)
        return shifted

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device
        schedule = self._schedule.to(device)

        # Track the start of generated tokens for each batch item
        start_list = []
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            n_masked = int((block_input_ids == self.mask_id).sum().item())
            start_list.append(self.block_size - n_masked)

        # Fast path: no mask tokens → STAGING_PREFILL
        mask_index_global = forward_batch.input_ids == self.mask_id
        if not mask_index_global.any():
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            if self.apply_logit_shift:
                full_logits = out.logits_output.full_logits
                prefix_lens = forward_batch.extend_prefix_lens
                offset = 0
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
                    offset += n_new
            return out.logits_output, [], out.can_run_graph

        # Denoising loop
        logits_output = None
        can_run_cuda_graph = True

        for step in range(self.denoising_steps):
            mask_index_global = forward_batch.input_ids == self.mask_id
            if not mask_index_global.any():
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            raw_full_logits = logits_output.full_logits

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

                # Sample tokens (supports temperature/top_k/top_p)
                x0, x0_p = sample_tokens(
                    curr_logits,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                )

                # low_confidence_dynamic selection
                confidence = torch.where(
                    block_mask_index, x0_p, torch.tensor(-np.inf, device=device)
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

        # Commit pass: final forward to store KV and capture prev_last_logits
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        if self.apply_logit_shift:
            commit_full_logits = logits_output.full_logits
            for batch_id in range(batch_size):
                if start_list[batch_id] >= self.block_size:
                    continue
                req_pool_idx = int(forward_batch.req_pool_indices[batch_id].item())
                block_last_idx = (batch_id + 1) * self.block_size - 1
                self._prev_last_logits[req_pool_idx] = (
                    commit_full_logits[block_last_idx].detach().clone()
                )

        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i]:] for i in range(batch_size)
        ]

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = DreamShift
