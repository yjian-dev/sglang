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


try:
    from flashinfer.sampling import top_k_top_p_sampling_from_probs as _fi_sample
    _HAS_FLASHINFER_SAMPLING = True
except ImportError:
    _HAS_FLASHINFER_SAMPLING = False


def _batched_sample(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched sampling from [N, vocab_size] logits on GPU.

    Returns (token_ids [N], probs [N]) — both stay on GPU, no .item() sync.
    Uses flashinfer fused kernel when available (single kernel for topk+topp+sample).
    """
    if temperature <= 0:
        probs = F.softmax(logits, dim=-1)
        token_ids = probs.argmax(dim=-1)
        token_probs = probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        return token_ids, token_probs

    scaled = logits if temperature == 1.0 else logits / temperature
    probs = F.softmax(scaled, dim=-1)

    if _HAS_FLASHINFER_SAMPLING:
        n = probs.shape[0]
        top_ks = torch.full((n,), top_k, dtype=torch.int32, device=probs.device)
        top_ps = torch.full((n,), top_p, dtype=torch.float32, device=probs.device)
        token_ids = _fi_sample(
            probs.contiguous(), top_ks, top_ps, filter_apply_order="joint"
        )
        token_probs = probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        return token_ids, token_probs

    # Fallback: manual implementation
    if top_k > 0:
        topk_vals, _ = scaled.topk(top_k, dim=-1)
        scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = scaled.sort(dim=-1, descending=True)
        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        mask = (cum_probs - sorted_logits.softmax(dim=-1)) >= top_p
        sorted_logits[mask] = float("-inf")
        scaled = sorted_logits.scatter(1, sorted_idx, sorted_logits)
    probs = F.softmax(scaled, dim=-1)
    token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
    token_probs = probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    return token_ids, token_probs


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

        self.use_spec_verify: bool = config.algorithm_config.get(
            "use_spec_verify", False
        )

        # Per-request state (keyed by req_pool_idx)
        self._prev_last_logits: Dict[int, torch.Tensor] = {}
        self._carry: Dict[int, int] = {}
        self._pending: Dict[int, int] = {}
        # Force next Case C to use this token instead of sampling
        self._force_next_token: Dict[int, int] = {}
        # Spec verification: draft distribution q(x) for the accepted t_carry
        self._pending_draft_probs: Dict[int, torch.Tensor] = {}

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

        # Determine per-request: prefill (no MASK) vs decode (has MASK)
        # Support mixed batches where some requests are prefill and others decode.
        extend_lens = forward_batch.extend_seq_lens  # tokens per request
        req_pool_indices_cpu = forward_batch.req_pool_indices[:batch_size].tolist()

        # Check per-request MASK presence
        is_decode = []  # bool per bid
        offset = 0
        for bid in range(batch_size):
            n_tokens = int(extend_lens[bid]) if extend_lens is not None else int(forward_batch.seq_lens[bid].item())
            chunk = forward_batch.input_ids[offset:offset + n_tokens]
            is_decode.append((chunk == self.mask_id).any().item())
            offset += n_tokens

        has_any_decode = any(is_decode)
        has_any_prefill = not all(is_decode)

        # ── Pure prefill (no decode requests) ──────────────────────────
        if not has_any_decode:
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
                req_pool_idx = req_pool_indices_cpu[bid]
                last_idx = offset + n_new - 1
                self._prev_last_logits[req_pool_idx] = (
                    full_logits[last_idx].detach().clone()
                )
                offset += n_new

            self._stats["total_forwards"] += 1
            return out.logits_output, [], out.can_run_graph

        # ── Decode ─────────────────────────────────────────────────────
        import time as _t; _t0 = _t.perf_counter()
        blk = self.block_size
        self._dllm_write_override.clear()
        self._kv_trim_info.clear()
        self._advance_override.clear()

        # Build decode-only bid list (skip prefill requests in Phase 1/3)
        decode_bids = [bid for bid in range(batch_size) if is_decode[bid]]
        prefill_bids = [bid for bid in range(batch_size) if not is_decode[bid]]

        # Phase 1: Vectorized case determination + input_ids fill
        # Classify each request: case_code 0=A, 1=B, 2=C
        case_codes = []       # int per bid
        tok0s = []            # tok_0 per bid (int)
        tok1s = []            # tok_1 per bid (int or None)
        pre_sample_logits = []
        pre_sample_bids = []
        pre_sample_cases = []  # 1=B, 2=C

        for bid in range(batch_size):
            rpx = req_pool_indices_cpu[bid]
            pending = self._pending.pop(rpx, None)
            carry = self._carry.pop(rpx, None)

            if pending is not None and carry is not None:
                # Case A
                forward_batch.input_ids[bid * blk] = pending
                forward_batch.input_ids[bid * blk + 1] = carry
                case_codes.append(0)
                tok0s.append(pending)
                tok1s.append(carry)
            elif pending is not None:
                # Case B — need to sample fresh
                forward_batch.input_ids[bid * blk] = pending
                prev = self._prev_last_logits.get(rpx)
                if prev is not None:
                    pre_sample_logits.append(prev)
                    pre_sample_bids.append(bid)
                    pre_sample_cases.append(1)
                    tok1s.append(None)  # filled after sampling
                else:
                    forward_batch.input_ids[bid * blk + 1] = self.mask_id
                    tok1s.append(self.mask_id)
                case_codes.append(1)
                tok0s.append(pending)
            else:
                # Case C — use forced token (from spec reject) or sample t_0
                forced = self._force_next_token.pop(rpx, None)
                if forced is not None:
                    forward_batch.input_ids[bid * blk] = forced
                    tok0s.append(forced)
                else:
                    prev = self._prev_last_logits.get(rpx)
                    if prev is not None:
                        pre_sample_logits.append(prev)
                        pre_sample_bids.append(bid)
                        pre_sample_cases.append(2)
                        tok0s.append(None)  # filled after sampling
                    else:
                        forward_batch.input_ids[bid * blk] = self.mask_id
                        tok0s.append(self.mask_id)
                case_codes.append(2)
                tok1s.append(None)

        # Batched pre-forward sampling (single kernel + single sync)
        if pre_sample_logits:
            sampled, _ = _batched_sample(
                torch.stack(pre_sample_logits),
                self.temperature, self.top_k, self.top_p,
            )
            sampled_cpu = sampled.tolist()
            for i, bid in enumerate(pre_sample_bids):
                tok = sampled_cpu[i]
                if pre_sample_cases[i] == 1:  # B
                    forward_batch.input_ids[bid * blk + 1] = tok
                    tok1s[bid] = tok
                else:  # C
                    forward_batch.input_ids[bid * blk] = tok
                    tok0s[bid] = tok

        # Phase 2: Single forward
        _t1 = _t.perf_counter()
        forward_batch.dllm_force_causal = True
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        forward_batch.dllm_force_causal = False
        _t2 = _t.perf_counter()
        logits_output = out.logits_output
        full_logits = logits_output.full_logits

        # Phase 3: Vectorized post-forward

        # Build diff/carry index tensors (no Python loop — use tensor arithmetic)
        case_arr = torch.tensor(case_codes, dtype=torch.long, device=device)
        bases = torch.arange(batch_size, device=device) * blk
        # Case A/B: diff@base+1, carry@base+2;  Case C: diff@base+0, carry@base+1
        is_ab = (case_arr <= 1)
        diff_idx = torch.where(is_ab, bases + 1, bases)
        carry_idx = torch.where(is_ab, bases + 2, bases + 1)

        diff_logits = full_logits[diff_idx]
        carry_logits = full_logits[carry_idx]

        diff_ids, _ = _batched_sample(
            diff_logits, self.temperature, self.top_k, self.top_p
        )
        carry_ids, carry_probs = _batched_sample(
            carry_logits, self.temperature, self.top_k, self.top_p
        )

        # Single sync: all sampled data + seq_lens + KV indices to CPU
        diff_ids_cpu = diff_ids.tolist()
        carry_ids_cpu = carry_ids.tolist()
        carry_probs_cpu = carry_probs.tolist()
        seq_lens_cpu = forward_batch.seq_lens[:batch_size].tolist()

        # Spec verification for Case A (before trim, since it may change trim count)
        spec_rejected = set()
        if self.use_spec_verify and batch_size > 0:
            for bid in range(batch_size):
                cc = case_codes[bid]
                if cc != 0:  # Only Case A has both pending + carry to verify
                    continue
                rpx = req_pool_indices_cpu[bid]
                draft_probs = self._pending_draft_probs.pop(rpx, None)
                if draft_probs is None:
                    continue
                t1 = tok1s[bid]  # the carried token to verify
                # Clean distribution from position 0 (pending's hidden)
                base = bid * blk
                clean_logits_raw = full_logits[base + 0]
                if self.temperature > 0 and self.temperature != 1.0:
                    clean_logits_raw = clean_logits_raw / self.temperature
                clean_probs = F.softmax(clean_logits_raw, dim=-1)
                p_x = clean_probs[t1].item()
                q_x = draft_probs[t1].item()
                r = p_x / q_x if q_x > 0 else 0.0
                if r >= 1.0 or torch.rand(1, device=device).item() < r:
                    continue  # accept verified
                # REJECT: resample from corrected distribution max(0, p-q)
                corrected = torch.clamp(clean_probs - draft_probs, min=0)
                csum = corrected.sum()
                if csum > 0:
                    corrected = corrected / csum
                    new_carry = torch.multinomial(corrected, num_samples=1).item()
                else:
                    new_carry = torch.multinomial(clean_probs, num_samples=1).item()
                tok1s[bid] = new_carry
                # Clear carry state — this bid is now a reject
                spec_rejected.add(rpx)

        # Batch KV index lookup (after spec verify, which may change trim counts)
        req_to_token = model_runner.req_to_token_pool.req_to_token
        trim_req_indices = []
        trim_positions = []
        trim_counts_per_bid = []
        for bid in range(batch_size):
            rpx = req_pool_indices_cpu[bid]
            sl = int(seq_lens_cpu[bid])
            if rpx in spec_rejected:
                tc = 1  # spec reject: free MASK only (carry KV overwritten next round)
            elif case_codes[bid] <= 1:
                tc = 1  # A/B: trim MASK only
            else:
                tc = 2  # C: trim 2 MASKs
            trim_counts_per_bid.append(tc)
            for t in range(tc):
                trim_req_indices.append(rpx)
                trim_positions.append(sl - 1 - t)
        if trim_req_indices:
            kv_indices_all = req_to_token[trim_req_indices, trim_positions].tolist()
        else:
            kv_indices_all = []

        # Process accept/reject
        next_token_ids_list = []
        trim_offset = 0

        for bid in range(batch_size):
            rpx = req_pool_indices_cpu[bid]
            t_diff = diff_ids_cpu[bid]
            t_carry = carry_ids_cpu[bid]
            accepted = carry_probs_cpu[bid] >= self.confidence_threshold
            cc = case_codes[bid]
            t0 = tok0s[bid]
            t1 = tok1s[bid]

            # Spec verification may override Case A accept
            if rpx in spec_rejected:
                # Spec rejected: output corrected carry (1 token). DON'T set
                # as pending — instead force next Case C to use it as t_0,
                # where it will get FRESH clean KV. This avoids double-output
                # AND avoids wasting a forward (deferred approach wastes 1).
                output_tokens = [t1]  # corrected carry, 1 token output
                dllm_tokens = [t0, t1, self.mask_id]
                self._prev_last_logits[rpx] = full_logits[bid * blk + 0]
                self._force_next_token[rpx] = t1  # next Case C uses this
                self._pending_draft_probs.pop(rpx, None)
                self._stats["reject_count"] += 1
                advance = 2   # pending + corrected committed
                trim_count = 1  # only MASK trimmed from kv_committed_len
            elif cc <= 1:  # A or B (normal path)
                if accepted:
                    output_tokens = [t1, t_diff]
                    dllm_tokens = [t0, t1, t_diff]
                    self._pending[rpx] = t_diff
                    self._carry[rpx] = t_carry
                    self._prev_last_logits[rpx] = carry_logits[bid]
                    # Store draft probs for spec verification in next round
                    if self.use_spec_verify:
                        carry_logits_raw = carry_logits[bid]
                        if self.temperature > 0 and self.temperature != 1.0:
                            carry_logits_raw = carry_logits_raw / self.temperature
                        self._pending_draft_probs[rpx] = F.softmax(
                            carry_logits_raw, dim=-1
                        )
                    self._stats["accept_count"] += 1
                else:
                    output_tokens = [t1]
                    dllm_tokens = [t0, t1, self.mask_id]
                    self._prev_last_logits[rpx] = diff_logits[bid]
                    self._pending_draft_probs.pop(rpx, None)
                    self._stats["reject_count"] += 1
                advance = 2
                trim_count = 1
            else:  # C
                if accepted:
                    output_tokens = [t0, t_diff]
                    dllm_tokens = [t0, t_diff, self.mask_id]
                    self._pending[rpx] = t_diff
                    self._carry[rpx] = t_carry
                    self._prev_last_logits[rpx] = carry_logits[bid]
                    if self.use_spec_verify:
                        carry_logits_raw = carry_logits[bid]
                        if self.temperature > 0 and self.temperature != 1.0:
                            carry_logits_raw = carry_logits_raw / self.temperature
                        self._pending_draft_probs[rpx] = F.softmax(
                            carry_logits_raw, dim=-1
                        )
                    self._stats["accept_count"] += 1
                else:
                    output_tokens = [t0]
                    dllm_tokens = [t0, self.mask_id, self.mask_id]
                    self._prev_last_logits[rpx] = diff_logits[bid]
                    self._pending_draft_probs.pop(rpx, None)
                    self._stats["reject_count"] += 1
                advance = 1
                trim_count = 2

            next_token_ids_list.append(
                torch.tensor(output_tokens, device=device)
            )
            self._dllm_write_override[rpx] = dllm_tokens
            self._advance_override[rpx] = advance

            actual_trim = trim_counts_per_bid[bid]
            kv_indices_to_free = kv_indices_all[trim_offset:trim_offset + actual_trim]
            trim_offset += actual_trim
            self._kv_trim_info[rpx] = {
                "kv_indices": kv_indices_to_free,
                "trim_count": trim_count,
            }

        # Debug: log every decode step for single-request tracing
        if batch_size == 1:
            rpx0 = req_pool_indices_cpu[0]
            cc0 = case_codes[0]
            case_name = {0:"A", 1:"B", 2:"C"}.get(cc0, "?")
            sr = " SPEC_REJ" if rpx0 in spec_rejected else ""
            out_toks = next_token_ids_list[0].tolist() if next_token_ids_list else []
            accepted_str = "acc" if carry_probs_cpu[0] >= self.confidence_threshold else "rej"
            logger.info(
                f"[STEP] {case_name}{sr} "
                f"in=[{tok0s[0]},{tok1s[0]}] diff={diff_ids_cpu[0]} carry={carry_ids_cpu[0]}({accepted_str}) "
                f"→ out={out_toks} adv={self._advance_override.get(rpx0,'?')} trim={trim_counts_per_bid[0]} "
                f"next: pend={self._pending.get(rpx0,'∅')} carry={self._carry.get(rpx0,'∅')}"
            )

        # Stats
        _t3 = _t.perf_counter()
        self._stats["total_forwards"] += 1
        self._stats["total_tokens"] += sum(len(t) for t in next_token_ids_list)
        self._stats.setdefault("p1", 0.0); self._stats["p1"] += (_t1-_t0)*1000
        self._stats.setdefault("p2", 0.0); self._stats["p2"] += (_t2-_t1)*1000
        self._stats.setdefault("p3", 0.0); self._stats["p3"] += (_t3-_t2)*1000
        n_spec_rej = len(spec_rejected)
        if self._stats["total_forwards"] % 500 == 0 or n_spec_rej > 0:
            s = self._stats
            tok_per_fwd = s["total_tokens"] / max(s["total_forwards"], 1)
            total_decisions = s["accept_count"] + s["reject_count"]
            accept_rate = (
                s["accept_count"] / total_decisions * 100
                if total_decisions > 0
                else 0
            )
            n = s["total_forwards"]
            logger.info(
                f"[DreamShiftBlock2] fwd={n}, "
                f"tok/fwd={tok_per_fwd:.2f}, "
                f"accept={accept_rate:.1f}%, "
                f"p1={s['p1']/n:.2f} p2={s['p2']/n:.2f} p3={s['p3']/n:.2f}ms"
            )

        return logits_output, next_token_ids_list, out.can_run_graph


Algorithm = DreamShiftBlock2
