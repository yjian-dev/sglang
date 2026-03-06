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

import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

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
        self.remasking_strategy: str = config.algorithm_config.get(
            "remasking_strategy", "low_confidence_dynamic"
        )
        self._schedule = get_num_transfer_tokens(self.block_size, self.denoising_steps)
        self.apply_logit_shift: bool = config.causal_prefill
        self._prev_last_logits: Dict[int, torch.Tensor] = {}
        self.sharp_logit_shift: bool = config.algorithm_config.get(
            "sharp_logit_shift", True
        )
        self.adaptive_schedule: bool = config.algorithm_config.get(
            "adaptive_schedule", False
        )
        self.stream_batch: bool = config.algorithm_config.get(
            "stream_batch", False
        )
        self._denoising_rids: set = set()  # request IDs currently denoising
        self._profile_stats: Dict[str, float] = {
            'total_blocks': 0, 'all_confident_step0': 0,
            'steps_used': 0, 'total_calls': 0,
            'total_forwards': 0, 'total_tokens_generated': 0,
            'total_block_forward_slots': 0,  # total block*forward opportunities
            'wasted_block_forward_slots': 0,  # slots where block was already done
        }

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
            if self.sharp_logit_shift:
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
        actual_steps_used = 0
        n_forwards = 0

        if self.stream_batch:
            # Stream Batch: single forward per run() call
            return self._run_stream_batch_step(
                model_runner, forward_batch, batch_size, device, schedule, start_list
            )

        for step in range(self.denoising_steps):
            mask_index_global = forward_batch.input_ids == self.mask_id
            if not mask_index_global.any():
                break
            actual_steps_used = step + 1

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            n_forwards += 1
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

                # Token transfer selection
                confidence = torch.where(
                    block_mask_index, x0_p, torch.tensor(-np.inf, device=device)
                )
                high_conf_mask = confidence > self.threshold
                n_high = int(high_conf_mask.sum().item())
                n_masks = int(block_mask_index.sum().item())

                # Profiling
                if step == 0:
                    self._profile_stats['total_blocks'] += 1
                    self._profile_stats['all_confident_step0'] += (
                        1 if n_high >= n_masks else 0
                    )

                if self.adaptive_schedule and n_high >= n_masks:
                    # Adaptive: all positions confident → unmask all at once
                    transfer_index = high_conf_mask
                else:
                    # Standard schedule
                    num_to_transfer = min(n_transfer, n_masks)
                    if num_to_transfer == 0:
                        continue

                    if n_high >= num_to_transfer:
                        transfer_index = high_conf_mask
                    elif self.remasking_strategy == "low_confidence_causal":
                        transfer_index = torch.zeros_like(block_mask_index)
                        first_mask = block_mask_index.nonzero(as_tuple=True)[0][0].item()
                        transfer_index[first_mask] = True
                    else:
                        transfer_index = torch.zeros_like(block_mask_index)
                        _, idx = torch.topk(confidence, num_to_transfer)
                        transfer_index[idx] = True

                block_input_ids[transfer_index] = x0[transfer_index]

            # Stream Batch profiling: count done blocks after this step
            n_done_blocks = 0
            for bid in range(batch_size):
                bs = bid * self.block_size
                be = bs + self.block_size
                if not (forward_batch.input_ids[bs:be] == self.mask_id).any():
                    n_done_blocks += 1
            remaining_steps = self.denoising_steps - step - 1  # steps left after this
            self._profile_stats['total_block_forward_slots'] += batch_size
            self._profile_stats['wasted_block_forward_slots'] += n_done_blocks * remaining_steps / max(1, self.denoising_steps)

        # Commit pass
        forward_batch.dllm_is_commit = True
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        forward_batch.dllm_is_commit = False
        n_forwards += 1
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

        # Profiling stats
        n_tokens_gen = sum(
            self.block_size - s for s in start_list
        )
        self._profile_stats['steps_used'] += actual_steps_used
        self._profile_stats['total_calls'] += 1
        self._profile_stats['total_forwards'] += n_forwards
        self._profile_stats['total_tokens_generated'] += n_tokens_gen
        if self._profile_stats['total_calls'] % 500 == 0:
            s = self._profile_stats
            avg_fwd = s['total_forwards'] / s['total_calls']
            tok_per_fwd = s['total_tokens_generated'] / max(s['total_forwards'], 1)
            wasted = s.get('wasted_block_forward_slots', 0)
            total_slots = max(s.get('total_block_forward_slots', 1), 1)
            logger.info(
                f"[DreamShift Profile] calls={s['total_calls']}, "
                f"avg_steps={s['steps_used']/s['total_calls']:.2f}, "
                f"avg_forwards/block={avg_fwd:.2f}, "
                f"tokens/forward={tok_per_fwd:.2f}, "
                f"all_confident_step0="
                f"{s.get('all_confident_step0',0)}/{s.get('total_blocks',1)} "
                f"({s.get('all_confident_step0',0)/max(s.get('total_blocks',1),1)*100:.1f}%), "
                f"stream_batch_savings={wasted/total_slots*100:.1f}%"
            )

        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i]:] for i in range(batch_size)
        ]

        return logits_output, next_token_ids_list, can_run_cuda_graph


    def _run_stream_batch_step(self, model_runner, forward_batch, batch_size, device, schedule, start_list):
        """Single-forward stream batch step. Called once per scheduler round."""
        mask_index_global = forward_batch.input_ids == self.mask_id
        has_masks = mask_index_global.any()

        if not has_masks:
            # All masks filled → commit pass
            forward_batch.dllm_is_commit = True
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            forward_batch.dllm_is_commit = False
            logits_output = out.logits_output
            can_run_cuda_graph = out.can_run_graph

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

            # Return actual tokens → scheduler advances to next block
            next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
            next_token_ids_list = [
                next_token_ids[i, start_list[i]:] for i in range(batch_size)
            ]

            # Mark requests as done denoising
            if hasattr(forward_batch, 'rids') and forward_batch.rids:
                for rid in forward_batch.rids:
                    self._denoising_rids.discard(rid)

            # Profiling
            self._profile_stats['total_forwards'] += 1
            self._profile_stats['total_calls'] += 1
            n_tokens = sum(self.block_size - s for s in start_list)
            self._profile_stats['total_tokens_generated'] += n_tokens
            self._profile_stats['steps_used'] += 0  # commit only

            return logits_output, next_token_ids_list, can_run_cuda_graph

        # Denoise step: 1 forward, unmask tokens
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output = out.logits_output
        can_run_cuda_graph = out.can_run_graph
        raw_full_logits = logits_output.full_logits

        n_transfer = int(schedule[0].item())  # always use schedule[0] for simplicity

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

            x0, x0_p = sample_tokens(
                curr_logits,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )

            confidence = torch.where(
                block_mask_index, x0_p, torch.tensor(-np.inf, device=device)
            )
            high_conf_mask = confidence > self.threshold
            n_high = int(high_conf_mask.sum().item())
            n_masks = int(block_mask_index.sum().item())

            # Adaptive: if all confident, unmask all at once
            if n_high >= n_masks:
                transfer_index = high_conf_mask
            else:
                num_to_transfer = min(n_transfer, n_masks)
                if num_to_transfer == 0:
                    continue
                if n_high >= num_to_transfer:
                    transfer_index = high_conf_mask
                elif self.remasking_strategy == "low_confidence_causal":
                    transfer_index = torch.zeros_like(block_mask_index)
                    first_mask = block_mask_index.nonzero(as_tuple=True)[0][0].item()
                    transfer_index[first_mask] = True
                else:
                    transfer_index = torch.zeros_like(block_mask_index)
                    _, idx = torch.topk(confidence, num_to_transfer)
                    transfer_index[idx] = True

            block_input_ids[transfer_index] = x0[transfer_index]

        # Still denoising → return empty next_token_ids to keep request in batch
        # Mark requests as still denoising (via rids)
        if hasattr(forward_batch, 'rids') and forward_batch.rids:
            for rid in forward_batch.rids:
                self._denoising_rids.add(rid)

        # Check if all masks now filled → next call will be commit
        still_has_masks = (forward_batch.input_ids == self.mask_id).any()

        # Return empty token lists so scheduler doesn't advance
        empty_list = [torch.tensor([], dtype=torch.long, device=device)] * batch_size

        # Profiling
        self._profile_stats['total_forwards'] += 1
        self._profile_stats['steps_used'] += 1

        if not still_has_masks:
            # All filled after this step → next call will commit
            # But we still return empty to not advance yet
            pass

        return logits_output, empty_list, can_run_cuda_graph


Algorithm = DreamShift


class DreamShiftStreamBatch(DreamShift):
    """Stream Batch variant: each run() does exactly 1 forward.
    
    The scheduler calls run() once per round. Denoising state is tracked
    on the request objects (dllm_denoise_step, dllm_needs_commit).
    
    This enables pipeline-style scheduling where different requests
    can be at different denoising steps in the same batch.
    """

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device

        # Track the start of generated tokens for each batch item
        start_list = []
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            n_masked = int((block_input_ids == self.mask_id).sum().item())
            start_list.append(self.block_size - n_masked)

        # Fast path: no mask tokens → STAGING_PREFILL (same as base)
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

        # === SINGLE FORWARD: either denoise or commit ===
        # Check if this batch should commit (all masks already filled)
        needs_denoise = mask_index_global.any()

        if needs_denoise:
            # ONE denoise step
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            raw_full_logits = logits_output.full_logits

            schedule = self._schedule.to(device)
            # Use per-request step from the first request (all should be same in current scheduler)
            # TODO: support mixed steps when scheduler is updated
            step = 0  # For now, use schedule[0] for all

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

                x0, x0_p = sample_tokens(
                    curr_logits,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                )

                confidence = torch.where(
                    block_mask_index, x0_p, torch.tensor(-np.inf, device=device)
                )
                high_conf_mask = confidence > self.threshold
                n_high = int(high_conf_mask.sum().item())
                n_masks = int(block_mask_index.sum().item())

                n_transfer = int(schedule[min(step, len(schedule)-1)].item())

                # Adaptive: if all confident, unmask all
                if n_high >= n_masks:
                    transfer_index = high_conf_mask
                else:
                    num_to_transfer = min(n_transfer, n_masks)
                    if num_to_transfer == 0:
                        continue
                    if n_high >= num_to_transfer:
                        transfer_index = high_conf_mask
                    elif self.remasking_strategy == "low_confidence_causal":
                        transfer_index = torch.zeros_like(block_mask_index)
                        first_mask = block_mask_index.nonzero(as_tuple=True)[0][0].item()
                        transfer_index[first_mask] = True
                    else:
                        transfer_index = torch.zeros_like(block_mask_index)
                        _, idx = torch.topk(confidence, num_to_transfer)
                        transfer_index[idx] = True

                block_input_ids[transfer_index] = x0[transfer_index]

            # Check if all masks are now filled → need commit next round
            mask_remaining = (forward_batch.input_ids == self.mask_id).any()
            if not mask_remaining:
                # All filled → do commit in the SAME call (to avoid extra round trip)
                pass  # Fall through to commit below
            else:
                # Still has masks → return partial result, scheduler calls again
                next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
                next_token_ids_list = [
                    next_token_ids[i, start_list[i]:] for i in range(batch_size)
                ]
                return logits_output, next_token_ids_list, can_run_cuda_graph

        # COMMIT pass
        forward_batch.dllm_is_commit = True
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        forward_batch.dllm_is_commit = False
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
