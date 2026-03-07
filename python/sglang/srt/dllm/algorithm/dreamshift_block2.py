"""
DreamShift Block-2 decoding: ~2 tokens per forward with KV trim.

Uses block_size=3 to match the reference causal_block2_generate_with_shift
pattern where each forward has 2-3 input tokens: [pending?, token, MASK].
MASK KV is always trimmed after the forward (never persists in cache).
Deferred tokens ("pending") get clean KV in the next round.

Config keys (passed via --dllm-algorithm-config YAML):
  block_size:             int,   MUST be 3
  confidence_threshold:   float, default 0.99
  temperature:            float, default 1.0
  top_k:                  int,   default 50
  top_p:                  float, default 0.95
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


class DreamShiftBlock2(DllmAlgorithm):
    """
    Block-2 speculative generation: 1 forward → 1-2 output tokens.

    Three cases per round (block_size=3 positions):
      A) pending + carry: [pending, carry, MASK] → accept: [carry, t_diff], reject: [carry]
      B) pending only:    [pending, fresh, MASK] → accept: [fresh, t_diff], reject: [fresh]
      C) no pending:      [t_0, MASK, MASK]      → accept: [t_0, t_diff],  reject: [t_0]

    MASK KV is always freed after the forward (never persists in cache).
    Deferred tokens (t_diff) get clean KV when re-processed as "pending"
    in the next round.

    Communicates with the output processor via instance dicts:
      _dllm_write_override: per-request tokens to write to dllm_ids
      _kv_trim_info: per-request KV pool indices to free
      _advance_override: per-request variable dllm_block_offset advance
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        assert self.block_size == 3, (
            f"DreamShiftBlock2 requires block_size=3, got {self.block_size}"
        )
        self.temperature: float = config.algorithm_config.get("temperature", 1.0)
        self.top_k: int = config.algorithm_config.get("top_k", 50)
        self.top_p: float = config.algorithm_config.get("top_p", 0.95)
        self.confidence_threshold: float = config.algorithm_config.get(
            "confidence_threshold", 0.99
        )

        # Per-request state (keyed by req_pool_idx)
        self._prev_last_logits: Dict[int, torch.Tensor] = {}
        self._carry: Dict[int, int] = {}
        self._pending: Dict[int, int] = {}

        # Per-round signals to the output processor (read + cleared each round)
        self._dllm_write_override: Dict[int, List[int]] = {}
        self._kv_trim_info: Dict[int, dict] = {}
        self._advance_override: Dict[int, int] = {}

        self._stats = {
            "total_forwards": 0,
            "total_tokens": 0,
            "accept_count": 0,
            "reject_count": 0,
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

            self._stats["total_forwards"] += 1
            return out.logits_output, [], out.can_run_graph

        # ── Decode ─────────────────────────────────────────────────────
        bs = self.block_size
        self._dllm_write_override.clear()
        self._kv_trim_info.clear()
        self._advance_override.clear()

        # Phase 1: Determine case and fill positions
        next_token_ids_list = []
        case_types = []

        for bid in range(batch_size):
            req_pool_idx = int(forward_batch.req_pool_indices[bid].item())
            base = bid * bs

            has_pending = req_pool_idx in self._pending
            has_carry = req_pool_idx in self._carry

            if has_pending and has_carry:
                # Case A: [pending, carry, MASK]
                pending = self._pending.pop(req_pool_idx)
                carry = self._carry.pop(req_pool_idx)
                forward_batch.input_ids[base + 0] = pending
                forward_batch.input_ids[base + 1] = carry
                case_types.append(("A", pending, carry))

            elif has_pending:
                # Case B: [pending, fresh, MASK]
                pending = self._pending.pop(req_pool_idx)
                self._carry.pop(req_pool_idx, None)
                prev_logits = self._prev_last_logits.get(req_pool_idx)
                fresh = self.mask_id
                if prev_logits is not None:
                    fresh, _ = _sample(
                        prev_logits, self.temperature, self.top_k, self.top_p
                    )
                forward_batch.input_ids[base + 0] = pending
                forward_batch.input_ids[base + 1] = fresh
                case_types.append(("B", pending, fresh))

            else:
                # Case C: [t_0, MASK, MASK]
                self._carry.pop(req_pool_idx, None)
                prev_logits = self._prev_last_logits.get(req_pool_idx)
                t_0 = self.mask_id
                if prev_logits is not None:
                    t_0, _ = _sample(
                        prev_logits, self.temperature, self.top_k, self.top_p
                    )
                forward_batch.input_ids[base + 0] = t_0
                case_types.append(("C", t_0, None))

        # Phase 2: Single forward with causal attention within block
        forward_batch.dllm_force_causal = True
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        forward_batch.dllm_force_causal = False
        logits_output = out.logits_output
        full_logits = logits_output.full_logits

        # Phase 3: Sample tokens, accept/reject, manage KV trim
        req_to_token = model_runner.req_to_token_pool.req_to_token
        token_to_kv_pool = forward_batch.token_to_kv_pool
        num_layers = model_runner.model_config.num_hidden_layers

        for bid in range(batch_size):
            req_pool_idx = int(forward_batch.req_pool_indices[bid].item())
            seq_len = int(forward_batch.seq_lens[bid].item())
            base = bid * bs
            case, tok_0, tok_1 = case_types[bid]

            if case in ("A", "B"):
                logits_1 = full_logits[base + 1]
                logits_2 = full_logits[base + 2]

                t_diff, _ = _sample(
                    logits_1, self.temperature, self.top_k, self.top_p
                )
                t_carry, t_carry_prob = _sample(
                    logits_2, self.temperature, self.top_k, self.top_p
                )

                if t_carry_prob >= self.confidence_threshold:
                    output_tokens = [tok_1, t_diff]
                    dllm_tokens = [tok_0, tok_1, t_diff]
                    trim_count = 1
                    advance = 2
                    self._pending[req_pool_idx] = t_diff
                    self._carry[req_pool_idx] = t_carry
                    self._prev_last_logits[req_pool_idx] = logits_2.detach().clone()
                    self._stats["accept_count"] += 1
                else:
                    output_tokens = [tok_1]
                    dllm_tokens = [tok_0, tok_1, self.mask_id]
                    trim_count = 1
                    advance = 2
                    self._pending.pop(req_pool_idx, None)
                    self._prev_last_logits[req_pool_idx] = logits_1.detach().clone()
                    self._stats["reject_count"] += 1

            else:  # Case C
                logits_0 = full_logits[base + 0]
                logits_1 = full_logits[base + 1]

                t_diff, _ = _sample(
                    logits_0, self.temperature, self.top_k, self.top_p
                )
                t_carry, t_carry_prob = _sample(
                    logits_1, self.temperature, self.top_k, self.top_p
                )

                if t_carry_prob >= self.confidence_threshold:
                    output_tokens = [tok_0, t_diff]
                    dllm_tokens = [tok_0, t_diff, self.mask_id]
                    trim_count = 2
                    advance = 1
                    self._pending[req_pool_idx] = t_diff
                    self._carry[req_pool_idx] = t_carry
                    self._prev_last_logits[req_pool_idx] = logits_1.detach().clone()
                    self._stats["accept_count"] += 1
                else:
                    output_tokens = [tok_0]
                    dllm_tokens = [tok_0, self.mask_id, self.mask_id]
                    trim_count = 2
                    advance = 1
                    self._pending.pop(req_pool_idx, None)
                    self._prev_last_logits[req_pool_idx] = logits_0.detach().clone()
                    self._stats["reject_count"] += 1

            next_token_ids_list.append(
                torch.tensor(output_tokens, device=device)
            )
            self._dllm_write_override[req_pool_idx] = dllm_tokens
            self._advance_override[req_pool_idx] = advance

            # Collect and zero KV at MASK positions
            kv_indices_to_free = []
            for t in range(trim_count):
                pos = seq_len - 1 - t
                kv_idx = int(req_to_token[req_pool_idx, pos].item())
                kv_indices_to_free.append(kv_idx)
                for layer_id in range(num_layers):
                    k_buf, v_buf = token_to_kv_pool.get_kv_buffer(layer_id)
                    k_buf[kv_idx].zero_()
                    v_buf[kv_idx].zero_()
            self._kv_trim_info[req_pool_idx] = {
                "kv_indices": kv_indices_to_free,
                "trim_count": trim_count,
            }

        # Profiling
        self._stats["total_forwards"] += 1
        self._stats["total_tokens"] += sum(len(t) for t in next_token_ids_list)
        if self._stats["total_forwards"] % 500 == 0:
            s = self._stats
            tok_per_fwd = s["total_tokens"] / max(s["total_forwards"], 1)
            total_decisions = s["accept_count"] + s["reject_count"]
            accept_rate = (
                s["accept_count"] / total_decisions * 100
                if total_decisions > 0
                else 0
            )
            logger.info(
                f"[DreamShiftBlock2] fwd={s['total_forwards']}, "
                f"tok={s['total_tokens']}, "
                f"tok/fwd={tok_per_fwd:.2f}, "
                f"accept={accept_rate:.1f}%"
            )

        return logits_output, next_token_ids_list, out.can_run_graph


Algorithm = DreamShiftBlock2
