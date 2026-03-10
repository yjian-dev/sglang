"""
DreamShift Causal (block_size=1) dLLM decoding algorithm.

For SDAR models trained with block_size=1 and Dream-style token shift
(hidden[i] predicts token[i+1]).  With block_size=1, the model is
essentially autoregressive: the token at position P can be sampled from
the *previous* position's logits (prev_last_logits) before running the
forward pass.  This gives us 1 forward per token (AR-equivalent).

Flow:
  Prefill  → causal forward, save last logits as prev_last_logits
  Decode   → sample token from prev_last_logits, replace MASK,
             forward (writes clean KV), save new logits

Config keys (passed via --dllm-algorithm-config YAML):
  temperature:  float, default 1.0  (0 = greedy/argmax)
  top_k:        int,   default 0    (0 = disabled)
  top_p:        float, default 1.0  (1.0 = disabled)
"""

import logging
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Tuple[int, float]:
    """Sample a single token from 1-D logits. Returns (token_id, prob)."""
    if temperature <= 0:
        probs = F.softmax(logits, dim=-1)
        tok = probs.argmax().item()
        return tok, probs[tok].item()

    filtered = logits.clone()
    if top_k > 0:
        topk_vals = filtered.topk(top_k).values
        filtered[filtered < topk_vals[-1]] = -float("inf")
    if top_p < 1.0:
        sorted_logits, sorted_idx = filtered.sort(descending=True)
        cumsum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        mask = cumsum - sorted_logits.softmax(dim=-1) >= top_p
        sorted_logits[mask] = -float("inf")
        filtered = sorted_logits.scatter(0, sorted_idx, sorted_logits)
    if temperature != 1.0:
        filtered = filtered / temperature

    probs = F.softmax(filtered, dim=-1)
    tok = torch.multinomial(probs, num_samples=1).item()
    return tok, probs[tok].item()


class DreamShiftCausal(DllmAlgorithm):
    """
    Dream-shift decoding for block_size=1 SDAR models.

    Each decode step uses exactly 1 model forward, matching AR throughput.
    The token is pre-sampled from prev_last_logits (dream shift property),
    then forwarded to produce clean KV and next-round logits.
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        assert self.block_size == 1, (
            f"DreamShiftCausal requires block_size=1, got {self.block_size}"
        )
        self.temperature: float = config.algorithm_config.get("temperature", 1.0)
        self.top_k: int = config.algorithm_config.get("top_k", 0)
        self.top_p: float = config.algorithm_config.get("top_p", 1.0)
        # per-request state: req_pool_idx → logits tensor (vocab_size,)
        self._prev_last_logits: Dict[int, torch.Tensor] = {}
        self._profile_stats: Dict[str, int] = {
            "total_forwards": 0,
            "total_tokens": 0,
        }

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device

        has_mask = (forward_batch.input_ids == self.mask_id).any().item()

        # ── Prefill (no MASK tokens) ──────────────────────────────────
        if not has_mask:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            full_logits = out.logits_output.full_logits

            # Save the last logit per request (dream shift: will predict
            # the first generated token).
            prefix_lens = forward_batch.extend_prefix_lens
            offset = 0
            for bid in range(batch_size):
                seq_len = int(forward_batch.seq_lens[bid].item())
                prefix_len = (
                    int(prefix_lens[bid].item()) if prefix_lens is not None else 0
                )
                n_new = seq_len - prefix_len
                req_pool_idx = int(forward_batch.req_pool_indices[bid].item())
                last_idx = offset + n_new - 1
                self._prev_last_logits[req_pool_idx] = (
                    full_logits[last_idx].detach().clone()
                )
                offset += n_new

            self._profile_stats["total_forwards"] += 1
            return out.logits_output, [], out.can_run_graph

        # ── Decode (block_size=1: each request has exactly 1 MASK) ────
        next_token_ids_list = []
        for bid in range(batch_size):
            req_pool_idx = int(forward_batch.req_pool_indices[bid].item())
            prev_logits = self._prev_last_logits.get(req_pool_idx)

            if prev_logits is None:
                # Fallback: should not happen in normal flow
                logger.warning(
                    f"No prev_last_logits for req_pool_idx={req_pool_idx}, "
                    "using MASK logits"
                )
                token_id = self.mask_id
            else:
                token_id, _ = _sample(
                    prev_logits, self.temperature, self.top_k, self.top_p
                )

            # Replace MASK with the sampled token
            forward_batch.input_ids[bid] = token_id
            next_token_ids_list.append(
                torch.tensor([token_id], device=device)
            )

        # Single forward with clean tokens → correct KV + next logits.
        # No need for dllm_is_commit: with 1 token the attention mode
        # (causal vs bidirectional) is identical.
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output = out.logits_output
        full_logits = logits_output.full_logits

        # Save logits for next round
        for bid in range(batch_size):
            req_pool_idx = int(forward_batch.req_pool_indices[bid].item())
            self._prev_last_logits[req_pool_idx] = (
                full_logits[bid].detach().clone()
            )

        self._profile_stats["total_forwards"] += 1
        self._profile_stats["total_tokens"] += batch_size
        if self._profile_stats["total_tokens"] % 1000 < batch_size:
            s = self._profile_stats
            logger.info(
                f"[DreamShiftCausal] forwards={s['total_forwards']}, "
                f"tokens={s['total_tokens']}, "
                f"tok/fwd={s['total_tokens']/max(s['total_forwards'],1):.2f}"
            )

        return logits_output, next_token_ids_list, out.can_run_graph


Algorithm = DreamShiftCausal
