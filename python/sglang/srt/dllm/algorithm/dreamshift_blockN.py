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


# Pre-allocated sampling parameter tensors (lazily initialized per device)
_SAMPLE_BUFS: Dict[torch.device, Dict[str, torch.Tensor]] = {}


def _get_sample_bufs(n: int, top_k: int, top_p: float, device: torch.device):
    """Get or create pre-allocated top_k/top_p tensors for flashinfer sampling."""
    bufs = _SAMPLE_BUFS.get(device)
    if bufs is None or bufs["size"] < n:
        new_size = max(n, 128)
        bufs = {
            "size": new_size,
            "top_ks": torch.full((new_size,), top_k, dtype=torch.int32, device=device),
            "top_ps": torch.full((new_size,), top_p, dtype=torch.float32, device=device),
        }
        _SAMPLE_BUFS[device] = bufs
    return bufs["top_ks"][:n], bufs["top_ps"][:n]


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
        top_ks, top_ps = _get_sample_bufs(n, top_k, top_p, probs.device)
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
      _kv_trim_info: per-request KV pool indices to free (GPU tensor)
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
        self._pending: Dict[int, int] = {}
        self._spec_tokens: Dict[int, List[int]] = {}
        # Store only scalar q(x) per spec token instead of full vocab distributions.
        # Each entry is a list of (token_id, q_prob) tuples.
        self._spec_draft_qx: Dict[int, List[float]] = {}
        self._force_next_token: Dict[int, int] = {}

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

    def cleanup_request(self, req_pool_idx: int):
        """Remove all per-request state for a finished request.

        Must be called when a request finishes to prevent stale state from
        being picked up by a new request that reuses the same req_pool_idx.
        """
        self._prev_last_logits.pop(req_pool_idx, None)
        self._pending.pop(req_pool_idx, None)
        self._spec_tokens.pop(req_pool_idx, None)
        self._spec_draft_qx.pop(req_pool_idx, None)
        self._force_next_token.pop(req_pool_idx, None)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        overlap_fn=None,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device
        blk = self.block_size  # 2*N - 1

        # Compute per-request extend lengths and cumulative offsets
        # for mixed decode+prefill batches (variable-length ragged layout)
        _el = forward_batch.extend_seq_lens
        if _el is not None:
            extend_lens_cpu = (
                _el.tolist()
                if isinstance(_el, torch.Tensor)
                else list(_el)
            )
        else:
            extend_lens_cpu = [blk] * batch_size
        # Cumulative offsets: base[bid] = sum(extend_lens[:bid])
        base_offsets = [0] * batch_size
        for bid in range(1, batch_size):
            base_offsets[bid] = base_offsets[bid - 1] + extend_lens_cpu[bid - 1]

        # Detect inline prefill requests (extend_len != blk)
        is_prefill = [extend_lens_cpu[bid] != blk for bid in range(batch_size)]

        # Determine if batch has any MASK tokens (decode vs prefill) — one GPU op
        req_pool_indices_cpu = forward_batch.req_pool_indices[:batch_size].tolist()
        has_any_decode = (forward_batch.input_ids == self.mask_id).any().item()

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

        # ── Decode (possibly mixed with inline prefill) ───────────────
        self._dllm_write_override.clear()
        self._kv_trim_info.clear()
        self._advance_override.clear()

        num_masks = self.num_masks  # N - 1

        # Phase 1: Classify requests and fill input_ids
        case_types = []
        t0_tokens = []
        was_forced = [False] * batch_size
        old_specs = [None] * batch_size
        old_draft_qx = [None] * batch_size  # scalar q(x) per spec

        pre_sample_logits = []
        pre_sample_bids = []

        for bid in range(batch_size):
            rpx = req_pool_indices_cpu[bid]

            # Skip inline prefill requests — they don't participate in decode
            if is_prefill[bid]:
                case_types.append('P')  # prefill
                t0_tokens.append(None)
                continue

            pending = self._pending.pop(rpx, None)
            specs = self._spec_tokens.pop(rpx, None)
            draft_qx = self._spec_draft_qx.pop(rpx, None)
            base = base_offsets[bid]

            if pending is not None and specs:
                forward_batch.input_ids[base + 0] = pending
                for si, sv in enumerate(specs):
                    forward_batch.input_ids[base + 1 + si] = sv
                case_types.append('V')
                t0_tokens.append(pending)
                old_specs[bid] = specs
                old_draft_qx[bid] = draft_qx
            else:
                forced = self._force_next_token.pop(rpx, None)
                if forced is not None:
                    forward_batch.input_ids[base] = forced
                    t0_tokens.append(forced)
                    was_forced[bid] = True
                else:
                    prev = self._prev_last_logits.get(rpx)
                    if prev is not None:
                        pre_sample_logits.append(prev)
                        pre_sample_bids.append(bid)
                        t0_tokens.append(None)
                    else:
                        forward_batch.input_ids[base] = self.mask_id
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
                forward_batch.input_ids[base_offsets[bid]] = tok
                t0_tokens[bid] = tok

        # Phase 2: Forward (GPU async — returns before GPU finishes)
        forward_batch.dllm_force_causal = True
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        forward_batch.dllm_force_causal = False

        # ── Overlap window: GPU is still computing, run CPU callback ──
        if overlap_fn is not None:
            overlap_fn()

        # Phase 3 starts here — first access to full_logits blocks until GPU done
        logits_output = out.logits_output
        full_logits = logits_output.full_logits

        # Phase 3: Batched post-forward — verify + sample + trim
        seq_lens_cpu = forward_batch.seq_lens[:batch_size].tolist()
        req_to_token = model_runner.req_to_token_pool.req_to_token

        # Separate decode bids from prefill bids
        decode_bids = [bid for bid in range(batch_size) if not is_prefill[bid]]
        prefill_bids = [bid for bid in range(batch_size) if is_prefill[bid]]

        # ── Handle inline prefill requests ─────────────────────────
        # Save last logits for each prefill request (like pure prefill path)
        if prefill_bids:
            prefill_logit_indices = []
            for bid in prefill_bids:
                # Last token position in the ragged batch for this request
                last_idx = base_offsets[bid] + extend_lens_cpu[bid] - 1
                prefill_logit_indices.append(last_idx)
            prefill_idx_t = torch.tensor(
                prefill_logit_indices, dtype=torch.long, device=device
            )
            prefill_logits = full_logits[prefill_idx_t].detach().clone()
            for k, bid in enumerate(prefill_bids):
                rpx = req_pool_indices_cpu[bid]
                self._prev_last_logits[rpx] = prefill_logits[k]

        # ── Step 1: Batched spec verification (decode only) ────────
        verify_bids = [bid for bid in decode_bids if case_types[bid] == 'V']
        cold_bids = [bid for bid in decode_bids if case_types[bid] == 'C']

        reject_at = {}       # bid -> rejected spec index
        corrected = {}       # bid -> corrected token id

        if self.use_spec_verify and verify_bids:
            nv = len(verify_bids)
            all_gather_idx = []
            all_spec_vals = []
            all_draft_qx_vals = []
            for bid in verify_bids:
                base = base_offsets[bid]
                for si in range(num_masks):
                    all_gather_idx.append(base + si)
                    all_spec_vals.append(old_specs[bid][si])
                    all_draft_qx_vals.append(old_draft_qx[bid][si])

            all_clean_logits = full_logits[
                torch.tensor(all_gather_idx, dtype=torch.long, device=device)
            ]
            if self.temperature > 0 and self.temperature != 1.0:
                all_clean_logits = all_clean_logits / self.temperature
            all_clean_probs = F.softmax(all_clean_logits, dim=-1)
            all_spec_vals_t = torch.tensor(
                all_spec_vals, dtype=torch.long, device=device
            )

            all_p = all_clean_probs.gather(1, all_spec_vals_t.unsqueeze(1)).squeeze(1)
            all_q = torch.tensor(
                all_draft_qx_vals, dtype=torch.float32, device=device
            )
            all_ratios = torch.where(all_q > 0, all_p / all_q, torch.zeros_like(all_p))
            all_rands = torch.rand(nv * num_masks, device=device)
            all_accepted = (all_ratios >= 1.0) | (all_rands < all_ratios)

            all_accepted_cpu = all_accepted.tolist()

            reject_indices = []
            for k, bid in enumerate(verify_bids):
                base_k = k * num_masks
                for si in range(num_masks):
                    if not all_accepted_cpu[base_k + si]:
                        reject_at[bid] = si
                        reject_indices.append((bid, si, base_k + si))
                        break

            if reject_indices:
                corr_indices = [idx for _, _, idx in reject_indices]
                corr_probs = all_clean_probs[corr_indices]
                corr_tokens = torch.multinomial(corr_probs, num_samples=1).squeeze(1)
                corr_tokens_cpu = corr_tokens.tolist()
                for i, (bid, si, idx) in enumerate(reject_indices):
                    corrected[bid] = corr_tokens_cpu[i]

        # ── Step 2: Batched sampling (decode requests only) ────────
        accepted_verify_bids = [
            bid for bid in verify_bids if bid not in reject_at
        ]
        gs = 1 + num_masks

        sample_logit_indices = []
        sample_bid_roles = []

        for bid in accepted_verify_bids:
            n_specs = len(old_specs[bid])
            clean_idx = base_offsets[bid] + n_specs
            for j in range(gs):
                sample_logit_indices.append(clean_idx + j)
            sample_bid_roles.append(bid)

        for bid in cold_bids:
            base = base_offsets[bid]
            for j in range(gs):
                sample_logit_indices.append(base + j)
            sample_bid_roles.append(bid)

        sampled_results = {}
        draft_qx_map = {}
        if sample_logit_indices:
            all_logits = full_logits[
                torch.tensor(sample_logit_indices, dtype=torch.long, device=device)
            ]
            all_ids, all_probs = _batched_sample(
                all_logits, self.temperature, self.top_k, self.top_p
            )

            draft_indices = []
            for group_start in range(0, len(sample_logit_indices), gs):
                for m in range(num_masks):
                    draft_indices.append(group_start + 1 + m)

            all_ids_cpu = all_ids.tolist()
            draft_qx_vals = all_probs[draft_indices].tolist() if draft_indices else []

            offset = 0
            draft_offset = 0
            for bid in sample_bid_roles:
                sampled_results[bid] = all_ids_cpu[offset:offset + gs]
                draft_qx_map[bid] = draft_qx_vals[draft_offset:draft_offset + num_masks]
                offset += gs
                draft_offset += num_masks

        # ── Step 3: Batched KV trim index lookup (decode only) ─────
        trim_counts = {}
        advances = {}
        for bid in decode_bids:
            if bid in reject_at:
                si = reject_at[bid]
                n_remaining = len(old_specs[bid]) - si
                trim_counts[bid] = n_remaining + num_masks
                advances[bid] = 1 + si
            elif case_types[bid] == 'V':
                trim_counts[bid] = num_masks
                advances[bid] = 1 + len(old_specs[bid])
            else:
                trim_counts[bid] = blk - 1
                advances[bid] = 1

        all_trim_rpx = []
        all_trim_pos = []
        for bid in decode_bids:
            rpx = req_pool_indices_cpu[bid]
            sl = int(seq_lens_cpu[bid])
            tc = trim_counts[bid]
            for t in range(tc):
                all_trim_rpx.append(rpx)
                all_trim_pos.append(sl - 1 - t)

        all_kv_indices = None
        if all_trim_rpx:
            all_kv_indices = req_to_token[all_trim_rpx, all_trim_pos]

        # ── Step 4: Assemble outputs ──────────────────────────────
        next_token_ids_list = []
        kv_offset = 0

        # Batched gather of logits to save for decode requests
        _logit_save_indices = []
        _logit_save_bids = []
        for bid in decode_bids:
            base = base_offsets[bid]
            if bid in reject_at:
                _logit_save_indices.append(base + reject_at[bid])
            elif case_types[bid] == 'V':
                _logit_save_indices.append(base + len(old_specs[bid]))
            else:
                _logit_save_indices.append(base)
            _logit_save_bids.append(bid)

        _saved_logits = {}
        if _logit_save_indices:
            _idx_t = torch.tensor(_logit_save_indices, dtype=torch.long, device=device)
            _all_saved = full_logits[_idx_t].detach().clone()
            for k, bid in enumerate(_logit_save_bids):
                _saved_logits[bid] = _all_saved[k]

        for bid in range(batch_size):
            rpx = req_pool_indices_cpu[bid]

            # Prefill requests: no output tokens, handled above
            if is_prefill[bid]:
                next_token_ids_list.append([])
                continue

            tc = trim_counts[bid]
            adv = advances[bid]

            if bid in reject_at:
                si = reject_at[bid]
                specs = old_specs[bid]
                ct = corrected[bid]
                output_tokens = list(specs[:si]) + [ct]
                dllm_tokens = [t0_tokens[bid]] + list(specs[:si]) + [ct]
                dllm_tokens += [self.mask_id] * (blk - len(dllm_tokens))
                self._force_next_token[rpx] = ct
                self._prev_last_logits[rpx] = _saved_logits[bid]
                self._stats["reject_count"] += 1

            elif case_types[bid] == 'V':
                specs = old_specs[bid]
                sr = sampled_results[bid]
                clean_token = sr[0]
                new_spec_tokens = sr[1:]
                output_tokens = list(specs) + [clean_token]
                dllm_tokens = [t0_tokens[bid]] + list(specs) + [clean_token]
                dllm_tokens += [self.mask_id] * (blk - len(dllm_tokens))
                self._pending[rpx] = clean_token
                self._spec_tokens[rpx] = new_spec_tokens
                self._spec_draft_qx[rpx] = draft_qx_map[bid]
                self._prev_last_logits[rpx] = _saved_logits[bid]
                self._stats["accept_count"] += 1

            else:
                t0 = t0_tokens[bid]
                sr = sampled_results[bid]
                clean_token = sr[0]
                new_spec_tokens = sr[1:]
                if was_forced[bid]:
                    output_tokens = [clean_token]
                else:
                    output_tokens = [t0, clean_token]
                dllm_tokens = [t0, clean_token] + new_spec_tokens
                dllm_tokens += [self.mask_id] * (blk - len(dllm_tokens))
                self._pending[rpx] = clean_token
                self._spec_tokens[rpx] = new_spec_tokens
                self._spec_draft_qx[rpx] = draft_qx_map[bid]
                self._prev_last_logits[rpx] = _saved_logits[bid]

            next_token_ids_list.append(output_tokens)
            self._dllm_write_override[rpx] = dllm_tokens
            self._advance_override[rpx] = adv
            self._kv_trim_info[rpx] = {
                "kv_indices_gpu": all_kv_indices[kv_offset:kv_offset + tc] if all_kv_indices is not None else None,
                "trim_count": tc,
            }
            kv_offset += tc

        # Debug logging for single request
        if batch_size == 1 and not is_prefill[0]:
            rpx0 = req_pool_indices_cpu[0]
            ct = case_types[0]
            out_toks = next_token_ids_list[0] if next_token_ids_list else []
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
        self._stats["total_forwards"] += 1
        self._stats["total_tokens"] += sum(len(t) for t in next_token_ids_list)
        if self._stats["total_forwards"] % 500 == 0:
            s = self._stats
            n = s["total_forwards"]
            tok_per_fwd = s["total_tokens"] / max(n, 1)
            total_decisions = s["accept_count"] + s["reject_count"]
            accept_rate = (
                s["accept_count"] / total_decisions * 100
                if total_decisions > 0
                else 0
            )
            logger.info(
                f"[DreamShiftBlockN] N={self.gen_block_size}, fwd={n}, bs={batch_size}, "
                f"tok/fwd={tok_per_fwd:.2f}, accept={accept_rate:.1f}%"
            )

        return logits_output, next_token_ids_list, out.can_run_graph


Algorithm = DreamShiftBlockN
