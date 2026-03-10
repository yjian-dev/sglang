"""
DreamShift Block-N decoding: up to N tokens per forward with speculative verification.

Generalizes Block-2 to arbitrary N. Each forward has two modes:
  Cold start:  [t0, M, M, ..., M]        (1 + 2(N-1) positions, padded)
  Verify:      [pending, spec0..specK, M, M, ..., M]  (1 + K + (N-1) positions)

The SGLang block_size must be set to 2*N - 1 (the max input size for verify rounds).
N is controlled by the `gen_block_size` config key.

Speculative tokens are verified left-to-right using the standard spec decoding criterion:
    r = p(x) / q(x)
    if r >= 1: always accept
    if r <  1: accept with probability r, else resample from max(0, p-q)

Config keys (passed via --dllm-algorithm-config YAML):
  block_size:             int,   MUST be 2*gen_block_size - 1
  gen_block_size:         int,   tokens per step (1 clean + N-1 spec). Default 3.
  confidence_threshold:   float, default 0.0 (disabled, rely on spec verify)
  temperature:            float, default 1.0
  top_k:                  int,   default 50
  top_p:                  float, default 0.95
  use_spec_verify:        bool,  default true
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

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


class DreamShiftBlockN(DllmAlgorithm):
    """
    Block-N speculative generation: 1 forward → 1 to N output tokens.

    Two modes per round (blk = 2*N - 1 positions):
      Verify:     [pending, spec0, ..., spec(N-2), M, M, ..., M]
                  → verify specs left-to-right, then sample N new tokens
      Cold start: [t0, M, M, ..., M]
                  → sample 1 clean + (N-1) spec, hold specs for next verify

    MASK KV is always freed after the forward (never persists in cache).
    Rejected specs cause a fallback to cold start with the corrected token.

    Communicates with the output processor via instance dicts:
      _dllm_write_override: per-request tokens to write to dllm_ids
      _kv_trim_info: per-request KV pool indices to free
      _advance_override: per-request variable dllm_block_offset advance
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.gen_block_size: int = config.algorithm_config.get("gen_block_size", 3)
        self.num_masks: int = self.gen_block_size - 1  # N-1 speculative tokens
        expected_blk = 2 * self.gen_block_size - 1
        assert self.block_size == expected_blk, (
            f"DreamShiftBlockN with gen_block_size={self.gen_block_size} requires "
            f"block_size={expected_blk}, got {self.block_size}"
        )
        self.temperature: float = config.algorithm_config.get("temperature", 1.0)
        self.top_k: int = config.algorithm_config.get("top_k", 50)
        self.top_p: float = config.algorithm_config.get("top_p", 0.95)
        self.confidence_threshold: float = config.algorithm_config.get(
            "confidence_threshold", 0.0
        )
        self.use_spec_verify: bool = config.algorithm_config.get(
            "use_spec_verify", True
        )

        # Per-request state (keyed by req_pool_idx)
        self._prev_last_logits: Dict[int, torch.Tensor] = {}
        self._pending: Dict[int, int] = {}          # clean token awaiting clean KV
        self._spec_tokens: Dict[int, List[int]] = {}  # speculative token values
        self._spec_draft_probs: Dict[int, List[torch.Tensor]] = {}  # draft dists
        self._force_next_token: Dict[int, int] = {}  # corrected token on reject

        # Per-round signals to the output processor
        self._dllm_write_override: Dict[int, List[int]] = {}
        self._kv_trim_info: Dict[int, dict] = {}
        self._advance_override: Dict[int, int] = {}

        self._stats = {
            "total_forwards": 0,
            "total_tokens": 0,
            "accept_count": 0,
            "reject_count": 0,
        }
        logger.info(
            f"[DreamShiftBlockN] gen_block_size={self.gen_block_size}, "
            f"block_size={self.block_size}, num_masks={self.num_masks}, "
            f"spec_verify={self.use_spec_verify}"
        )

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device
        blk = self.block_size  # 2*N - 1

        # Determine per-request: prefill (no MASK) vs decode (has MASK)
        extend_lens = forward_batch.extend_seq_lens
        req_pool_indices_cpu = forward_batch.req_pool_indices[:batch_size].tolist()

        is_decode = []
        offset = 0
        for bid in range(batch_size):
            n_tokens = int(extend_lens[bid]) if extend_lens is not None else int(
                forward_batch.seq_lens[bid].item())
            chunk = forward_batch.input_ids[offset:offset + n_tokens]
            is_decode.append((chunk == self.mask_id).any().item())
            offset += n_tokens

        has_any_decode = any(is_decode)

        # ── Pure prefill ──────────────────────────────────────────────
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
                rpx = req_pool_indices_cpu[bid]
                last_idx = offset + n_new - 1
                self._prev_last_logits[rpx] = (
                    full_logits[last_idx].detach().clone()
                )
                offset += n_new

            self._stats["total_forwards"] += 1
            return out.logits_output, [], out.can_run_graph

        # ── Decode ────────────────────────────────────────────────────
        import time as _t; _t0 = _t.perf_counter()

        self._dllm_write_override.clear()
        self._kv_trim_info.clear()
        self._advance_override.clear()

        N = self.gen_block_size
        num_masks = self.num_masks  # N - 1

        # Phase 1: Classify requests and fill input_ids
        # case_type: 'V' = verify (has pending + specs), 'C' = cold start
        case_types = []
        t0_tokens = []          # input token at pos 0 per bid
        was_forced = [False] * batch_size
        old_specs = [None] * batch_size       # spec token values from prev round
        old_draft_probs = [None] * batch_size  # draft probs from prev round

        pre_sample_logits = []
        pre_sample_bids = []

        for bid in range(batch_size):
            rpx = req_pool_indices_cpu[bid]
            pending = self._pending.pop(rpx, None)
            specs = self._spec_tokens.pop(rpx, None)
            draft_probs = self._spec_draft_probs.pop(rpx, None)

            if pending is not None and specs:
                # VERIFY round: [pending, spec0, spec1, ..., M, M, ...]
                forward_batch.input_ids[bid * blk + 0] = pending
                for si, sv in enumerate(specs):
                    forward_batch.input_ids[bid * blk + 1 + si] = sv
                # Remaining positions (1 + len(specs) .. blk-1) stay as MASK
                case_types.append('V')
                t0_tokens.append(pending)
                old_specs[bid] = specs
                old_draft_probs[bid] = draft_probs
            else:
                # COLD START: [t0, M, M, ..., M]
                forced = self._force_next_token.pop(rpx, None)
                if forced is not None:
                    forward_batch.input_ids[bid * blk] = forced
                    t0_tokens.append(forced)
                    was_forced[bid] = True
                else:
                    prev = self._prev_last_logits.get(rpx)
                    if prev is not None:
                        pre_sample_logits.append(prev)
                        pre_sample_bids.append(bid)
                        t0_tokens.append(None)  # filled after sampling
                    else:
                        forward_batch.input_ids[bid * blk] = self.mask_id
                        t0_tokens.append(self.mask_id)
                case_types.append('C')

        # Batched pre-forward sampling for cold start t0
        if pre_sample_logits:
            sampled, _ = _batched_sample(
                torch.stack(pre_sample_logits),
                self.temperature, self.top_k, self.top_p,
            )
            sampled_cpu = sampled.tolist()
            for i, bid in enumerate(pre_sample_bids):
                tok = sampled_cpu[i]
                forward_batch.input_ids[bid * blk] = tok
                t0_tokens[bid] = tok

        # Phase 2: Forward
        _t1 = _t.perf_counter()
        forward_batch.dllm_force_causal = True
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        forward_batch.dllm_force_causal = False
        _t2 = _t.perf_counter()

        logits_output = out.logits_output
        full_logits = logits_output.full_logits

        # Phase 3: Post-forward — verify + sample
        seq_lens_cpu = forward_batch.seq_lens[:batch_size].tolist()
        req_to_token = model_runner.req_to_token_pool.req_to_token

        next_token_ids_list = []

        for bid in range(batch_size):
            rpx = req_pool_indices_cpu[bid]
            sl = int(seq_lens_cpu[bid])
            base = bid * blk

            if case_types[bid] == 'V':
                # ── VERIFY ROUND ──────────────────────────────────
                specs = old_specs[bid]
                drafts = old_draft_probs[bid]
                n_specs = len(specs)
                rejected_idx = -1
                corrected_token = None

                if self.use_spec_verify:
                    for si in range(n_specs):
                        spec_val = specs[si]
                        # Clean distribution from logits at position si
                        # si=0: pending's hidden → verifies spec[0]
                        # si=1: spec[0]'s hidden → verifies spec[1]
                        clean_logits_i = full_logits[base + si]
                        if self.temperature > 0 and self.temperature != 1.0:
                            clean_logits_i = clean_logits_i / self.temperature
                        clean_probs = F.softmax(clean_logits_i, dim=-1)
                        draft_probs_i = drafts[si]

                        p_x = clean_probs[spec_val].item()
                        q_x = draft_probs_i[spec_val].item()
                        r = p_x / q_x if q_x > 0 else 0.0

                        if r >= 1.0 or torch.rand(1, device=device).item() < r:
                            continue  # accepted

                        # REJECT at si
                        corrected = torch.clamp(clean_probs - draft_probs_i, min=0)
                        csum = corrected.sum()
                        if csum > 0:
                            corrected = corrected / csum
                            new_tok = torch.multinomial(corrected, num_samples=1).item()
                        else:
                            new_tok = torch.multinomial(clean_probs, num_samples=1).item()
                        rejected_idx = si
                        corrected_token = new_tok
                        break

                if rejected_idx >= 0:
                    # ── Spec rejected at index rejected_idx ──
                    n_accepted = rejected_idx
                    # Output: accepted specs + corrected token
                    output_tokens = list(specs[:n_accepted]) + [corrected_token]

                    # Trim: remaining specs + MASKs
                    n_remaining_specs = n_specs - rejected_idx
                    trim_count = n_remaining_specs + num_masks
                    advance = 1 + n_accepted  # pending + accepted specs

                    # Force corrected token as t0 in next cold start
                    self._force_next_token[rpx] = corrected_token
                    self._prev_last_logits[rpx] = full_logits[base + rejected_idx]

                    # Write dllm tokens: [pending, accepted_specs..., corrected, M...]
                    dllm_tokens = [t0_tokens[bid]] + list(specs[:n_accepted]) + [corrected_token]
                    dllm_tokens += [self.mask_id] * (blk - len(dllm_tokens))

                    self._stats["reject_count"] += 1
                else:
                    # ── All specs accepted ──
                    # Clean token from logits[n_specs] (last spec's position)
                    clean_logit_idx = base + n_specs

                    # Sample clean + new spec tokens
                    sample_indices = list(range(clean_logit_idx, clean_logit_idx + 1 + num_masks))
                    sample_logits = full_logits[sample_indices]
                    sampled_ids, sampled_probs = _batched_sample(
                        sample_logits, self.temperature, self.top_k, self.top_p
                    )
                    sampled_cpu = sampled_ids.tolist()
                    sampled_probs_cpu = sampled_probs.tolist()

                    clean_token = sampled_cpu[0]
                    new_spec_tokens = sampled_cpu[1:]
                    new_draft_probs_list = []
                    for m in range(num_masks):
                        d_logits = full_logits[clean_logit_idx + 1 + m]
                        if self.temperature > 0 and self.temperature != 1.0:
                            d_logits = d_logits / self.temperature
                        new_draft_probs_list.append(F.softmax(d_logits, dim=-1))

                    # Output: verified specs + clean token = N tokens
                    output_tokens = list(specs) + [clean_token]

                    # State for next round
                    self._pending[rpx] = clean_token
                    self._spec_tokens[rpx] = new_spec_tokens
                    self._spec_draft_probs[rpx] = new_draft_probs_list
                    self._prev_last_logits[rpx] = full_logits[clean_logit_idx]

                    # Trim MASKs only
                    trim_count = num_masks
                    advance = 1 + n_specs  # pending + all specs committed

                    # Write dllm tokens
                    dllm_tokens = [t0_tokens[bid]] + list(specs) + [clean_token]
                    dllm_tokens += [self.mask_id] * (blk - len(dllm_tokens))

                    self._stats["accept_count"] += 1

            else:
                # ── COLD START ────────────────────────────────────
                t0 = t0_tokens[bid]

                # Sample clean token from pos 0 + spec tokens from pos 1..num_masks
                sample_indices = list(range(base, base + 1 + num_masks))
                sample_logits = full_logits[sample_indices]
                sampled_ids, sampled_probs = _batched_sample(
                    sample_logits, self.temperature, self.top_k, self.top_p
                )
                sampled_cpu = sampled_ids.tolist()

                clean_token = sampled_cpu[0]
                new_spec_tokens = sampled_cpu[1:]
                new_draft_probs_list = []
                for m in range(num_masks):
                    d_logits = full_logits[base + 1 + m]
                    if self.temperature > 0 and self.temperature != 1.0:
                        d_logits = d_logits / self.temperature
                    new_draft_probs_list.append(F.softmax(d_logits, dim=-1))

                # Output: [t0, clean] if not forced, [clean] if forced
                if was_forced[bid]:
                    output_tokens = [clean_token]
                else:
                    output_tokens = [t0, clean_token]

                # State for next round
                self._pending[rpx] = clean_token
                self._spec_tokens[rpx] = new_spec_tokens
                self._spec_draft_probs[rpx] = new_draft_probs_list
                self._prev_last_logits[rpx] = full_logits[base]

                # Trim all positions except t0
                trim_count = blk - 1  # 2N-2
                advance = 1  # only t0 committed

                # Write dllm tokens
                dllm_tokens = [t0, clean_token] + new_spec_tokens
                dllm_tokens += [self.mask_id] * (blk - len(dllm_tokens))

            # Build KV trim info
            kv_indices = []
            for t in range(trim_count):
                kv_idx = req_to_token[rpx, sl - 1 - t].item()
                kv_indices.append(kv_idx)

            next_token_ids_list.append(
                torch.tensor(output_tokens, device=device)
            )
            self._dllm_write_override[rpx] = dllm_tokens
            self._advance_override[rpx] = advance
            self._kv_trim_info[rpx] = {
                "kv_indices": kv_indices,
                "trim_count": trim_count,
            }

        # Debug logging for single request
        if batch_size == 1:
            rpx0 = req_pool_indices_cpu[0]
            ct = case_types[0]
            out_toks = next_token_ids_list[0].tolist() if next_token_ids_list else []
            adv = self._advance_override.get(rpx0, '?')
            tc = self._kv_trim_info.get(rpx0, {}).get('trim_count', '?')
            n_pend = self._pending.get(rpx0, '∅')
            n_specs = self._spec_tokens.get(rpx0, [])
            logger.info(
                f"[STEP] {ct} t0={t0_tokens[0]} "
                f"→ out={out_toks} adv={adv} trim={tc} "
                f"next: pend={n_pend} specs={n_specs[:3]}..."
            )

        # Stats
        _t3 = _t.perf_counter()
        self._stats["total_forwards"] += 1
        self._stats["total_tokens"] += sum(len(t) for t in next_token_ids_list)
        self._stats.setdefault("p1", 0.0); self._stats["p1"] += (_t1 - _t0) * 1000
        self._stats.setdefault("p2", 0.0); self._stats["p2"] += (_t2 - _t1) * 1000
        self._stats.setdefault("p3", 0.0); self._stats["p3"] += (_t3 - _t2) * 1000

        if self._stats["total_forwards"] % 200 == 0:
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
                f"[DreamShiftBlockN] N={self.gen_block_size}, fwd={n}, "
                f"tok/fwd={tok_per_fwd:.2f}, "
                f"accept={accept_rate:.1f}%, "
                f"p1={s['p1']/n:.2f} p2={s['p2']/n:.2f} p3={s['p3']/n:.2f}ms"
            )

        return logits_output, next_token_ids_list, out.can_run_graph


Algorithm = DreamShiftBlockN
