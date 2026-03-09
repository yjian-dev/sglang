import argparse
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def top_k_logits(logits, k):
    if k <= 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool),
                                 -1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(mask_indices, float('-inf'))
    return logits


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]    # [batch, block]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)  # [batch*block, vocab]

    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    if temperature is not None and temperature <= 0:
        probs = F.softmax(logits, dim=-1)
        token = probs.argmax(dim=-1, keepdim=True)
        token_prob = torch.gather(probs, -1, token)
        return token.view(*orig_shape), token_prob.view(*orig_shape)
    if temperature != 1.0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)  # shape: [batch*block, vocab]
    assert probs.dim() == 2
    token = torch.multinomial(probs, num_samples=1)  # [batch*block, 1]
    token_prob = torch.gather(probs, -1, token)     # [batch*block, 1]

    return token.view(*orig_shape), token_prob.view(*orig_shape)


def get_num_transfer_tokens(block_length, steps):
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


@torch.no_grad()
def block_diffusion_generate(
        model,
        prompt,
        mask_id,
        gen_length=128,
        block_length=8,
        denoising_steps=8,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy='low_confidence_dynamic',
        confidence_threshold=0.85,
        eb_threshold=None,
        stopping_criteria_idx=None
    ):

    model.eval()
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length +
                  block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(
        num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                               .repeat_interleave(block_length, dim=1).unsqueeze(0)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    # Prefill stage
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:,
                                                       :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        model(cur_x,
              attention_mask=cur_attn_mask,
              position_ids=cur_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True)

    num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)

    # Decode stage
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
        ]
        cur_position_ids = position_ids[:, num_block *
                                        block_length:(num_block+1)*block_length]
        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                # Store kv cache
                model(cur_x,
                      attention_mask=cur_attn_mask,
                      position_ids=cur_position_ids,
                      past_key_values=past_key_values,
                      use_cache=True,
                      store_kv=True)
                break

            # Denosing
            logits = model(cur_x,
                           attention_mask=cur_attn_mask,
                           position_ids=cur_position_ids,
                           past_key_values=past_key_values,
                           use_cache=True,
                           store_kv=False).logits

            # Sampling
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # Sampling strategy
            if remasking_strategy == 'sequential':
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(cur_x.shape[0]):
                    if mask_index[j].any():
                        first_mask_index = mask_index[j].nonzero(as_tuple=True)[
                            0].min().item()
                        transfer_index[j, first_mask_index:first_mask_index +
                                       num_transfer_tokens[step]] = True
                    else:
                        raise ValueError(
                            "No mask tokens found in the current block.")

            elif remasking_strategy == 'low_confidence_static':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, idx = torch.topk(
                        confidence[j], num_transfer_tokens[step])
                    transfer_index[j, idx] = True

            elif remasking_strategy == 'low_confidence_dynamic':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    high_conf_mask = confidence[j] > confidence_threshold
                    num_high_confidence = high_conf_mask.sum()
                    if num_high_confidence >= num_transfer_tokens[step]:
                        transfer_index[j] = high_conf_mask
                    else:
                        _, idx = torch.topk(
                            confidence[j], num_transfer_tokens[step])
                        transfer_index[j, idx] = True
            elif remasking_strategy == "entropy_bounded":
                eps = 1e-12
                entropies = -(x0_p.clamp_min(eps) * (x0_p.clamp_min(eps)).log()).sum(dim=-1)
                entropies = torch.where(mask_index, entropies, torch.inf)
                ent_sorted, order = torch.sort(entropies, dim=1, descending=False)
                cumsum = torch.cumsum(ent_sorted, dim=1)
                for j in range(x0_p.shape[0]):
                    k = torch.searchsorted(cumsum[j], torch.tensor(eb_threshold, device=x0_p.device), right=False).item()
                    k = max(1, min(k, int(mask_index[j].sum().item())))
                    selected_token_indices = order[j, :k]
                    transfer_index[j, selected_token_indices] = True
                
            else:
                raise ValueError(
                    f"Unknown remasking strategy: {remasking_strategy}")

            cur_x[transfer_index] = x0[transfer_index]
        else:
            # for-else: executes if loop completed without break
            # This means we did denoising_steps iterations but never hit mask_index.sum()==0
            # We need to store KV for this block now
            model(cur_x,
                  attention_mask=cur_attn_mask,
                  position_ids=cur_position_ids,
                  past_key_values=past_key_values,
                  use_cache=True,
                  store_kv=True)

        x[:, num_block*block_length:(num_block+1)*block_length] = cur_x
        if stopping_criteria_idx is not None and any(stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx):
            break

    return x


@torch.no_grad()
def sliding_window_block_diffusion_generate(
        model,
        prompt,
        mask_id,
        gen_length=128,
        block_length=8,
        denoising_steps=8,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy='low_confidence_dynamic',
        confidence_threshold=0.85,
        eb_threshold=None,
        enable_revision=True,
        stopping_criteria_idx=None,
        device=None
    ):
    """
    Sliding window block diffusion with token revision (Ideas 2 + 3).
    
    Key differences from standard block_diffusion_generate:
    - Window slides by mini_block_length (half of block_length) instead of full block
    - Low-confidence tokens in the "revision zone" can be re-masked and re-decoded
    
    Args:
        enable_revision: If True, low-confidence tokens in revision zone are re-masked (Idea 3).
                        If False, only slides window without revision (Idea 2 only).
        device: The device to create tensors on. If None, inferred from model or input.
    """
    model.eval()
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()
    
    # Determine device - try input first, then model, then default to cuda
    if device is None:
        device = input_ids.device
        if device.type == 'cpu':
            # Try to get from model
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mini-block is half of block_length (slide step)
    mini_block_length = block_length // 2
    assert block_length % 2 == 0, "block_length must be even for sliding window"
    
    # Calculate total length in terms of mini-blocks
    num_mini_blocks = (prompt_length + gen_length + mini_block_length - 1) // mini_block_length
    # Ensure we have an even number of mini-blocks for complete windows
    if num_mini_blocks % 2 == 1:
        num_mini_blocks += 1
    total_length = num_mini_blocks * mini_block_length
    
    # Attention mask: block-causal at the mini-block level for committed tokens,
    # but the active window (2 mini-blocks) is bidirectional within itself
    # We'll build this dynamically as we slide
    position_ids = torch.arange(total_length, device=device).unsqueeze(0)
    
    # Initialize sequence with masks
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids.to(device)
    
    # Prefill: process complete mini-blocks from prompt
    prefill_mini_blocks = prompt_length // mini_block_length
    # Align to even number for window alignment
    if prefill_mini_blocks % 2 == 1:
        prefill_mini_blocks -= 1
    prefill_length = prefill_mini_blocks * mini_block_length
    
    # Prefill stage - commit prompt tokens to KV cache
    if prefill_length > 0:
        # Build attention mask for prefill: causal at mini-block level
        num_prefill_windows = prefill_mini_blocks // 2
        prefill_block_mask = torch.tril(torch.ones(
            num_prefill_windows, num_prefill_windows, device=device))
        prefill_attn_mask = prefill_block_mask.repeat_interleave(block_length, dim=0)\
                                              .repeat_interleave(block_length, dim=1).unsqueeze(0)
        
        cur_x = x[:, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        model(cur_x,
              attention_mask=prefill_attn_mask,
              position_ids=cur_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True)
    
    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps).to(device)
    
    # Track how many mini-blocks have been committed to KV cache
    committed_mini_blocks = prefill_mini_blocks
    
    # Track confidences for the "back" half of current window (will be revision zone next iteration)
    revision_zone_confidences = None
    revision_zone_tokens = None
    
    # Total mini-blocks to generate
    total_gen_mini_blocks = num_mini_blocks - prefill_mini_blocks
    
    # Process windows: each iteration processes one window and commits one mini-block
    # First window is special: both mini-blocks are masked, commit first mini-block
    # Subsequent windows: revision zone + new zone, commit revision zone
    
    window_idx = 0
    current_mini_block = prefill_mini_blocks  # Next mini-block to start window from
    
    while current_mini_block < num_mini_blocks - 1:  # Need at least 2 mini-blocks for a window
        window_start = current_mini_block * mini_block_length
        window_end = window_start + block_length  # Window is 2 mini-blocks = 1 block_length
        
        if window_end > total_length:
            break
            
        # Get current window tokens
        cur_x = x[:, window_start:window_end].clone()
        cur_position_ids = position_ids[:, window_start:window_end]
        
        # For denoising, we want full attention: query attends to all past KV + current window
        # The SDAR model uses flash_attn_func with causal=False when attention_mask is all True
        # This gives bidirectional attention which is what we want for block diffusion
        committed_length = committed_mini_blocks * mini_block_length
        full_attn_length = committed_length + block_length
        
        # Pass all-ones mask to trigger the "decoding" path in SDAR attention
        # which uses flash_attn_func with causal=False (full bidirectional)
        cur_attn_mask = torch.ones(1, block_length, full_attn_length, device=device, dtype=torch.bool)
        
        # Apply revision: re-mask low-confidence tokens in the first mini-block (revision zone)
        # Only applies after the first window (when we have revision_zone_confidences)
        if window_idx > 0 and enable_revision and revision_zone_confidences is not None:
            # Revision zone is the first mini-block of current window
            revision_start = 0
            revision_end = mini_block_length
            
            # Find low-confidence tokens to re-mask
            low_conf_mask = revision_zone_confidences < confidence_threshold
            
            # Re-mask those positions
            cur_x[:, revision_start:revision_end][low_conf_mask] = mask_id
        
        # Denoising loop for this window
        final_confidences = torch.zeros(1, block_length, device=device)
        
        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)
            
            if mask_index.sum() == 0:
                # All tokens unmasked - we're done with this window
                break
            
            # Forward pass (don't store KV yet)
            logits = model(cur_x,
                          attention_mask=cur_attn_mask,
                          position_ids=cur_position_ids,
                          past_key_values=past_key_values,
                          use_cache=True,
                          store_kv=False).logits
            
            # Sampling
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Update final confidences for all positions
            final_confidences = x0_p.clone()
            
            # Apply remasking strategy to select which tokens to unmask this step
            # On final step (step >= denoising_steps), transfer all remaining masked tokens
            is_final_step = step >= denoising_steps
            
            if remasking_strategy == 'sequential':
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(cur_x.shape[0]):
                    if mask_index[j].any():
                        first_mask_index = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                        if is_final_step:
                            transfer_index[j, first_mask_index:] = mask_index[j, first_mask_index:]
                        else:
                            transfer_index[j, first_mask_index:first_mask_index + num_transfer_tokens[step]] = True
                    
            elif remasking_strategy == 'low_confidence_static':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    if is_final_step:
                        transfer_index[j] = mask_index[j]
                    else:
                        num_to_transfer = min(num_transfer_tokens[step].item(), mask_index[j].sum().item())
                        if num_to_transfer > 0:
                            _, idx = torch.topk(confidence[j], num_to_transfer)
                            transfer_index[j, idx] = True
                    
            elif remasking_strategy == 'low_confidence_dynamic':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    if is_final_step:
                        transfer_index[j] = mask_index[j]
                    else:
                        high_conf_mask = confidence[j] > confidence_threshold
                        num_high_confidence = high_conf_mask.sum()
                        num_to_transfer = min(num_transfer_tokens[step].item(), mask_index[j].sum().item())
                        if num_to_transfer > 0:
                            if num_high_confidence >= num_to_transfer:
                                transfer_index[j] = high_conf_mask
                            else:
                                _, idx = torch.topk(confidence[j], num_to_transfer)
                                transfer_index[j, idx] = True
                            
            elif remasking_strategy == "entropy_bounded":
                eps = 1e-12
                entropies = -(x0_p.clamp_min(eps) * (x0_p.clamp_min(eps)).log()).sum(dim=-1)
                entropies = torch.where(mask_index, entropies, torch.inf)
                ent_sorted, order = torch.sort(entropies, dim=1, descending=False)
                cumsum = torch.cumsum(ent_sorted, dim=1)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(x0_p.shape[0]):
                    k = torch.searchsorted(cumsum[j], torch.tensor(eb_threshold, device=x0_p.device), right=False).item()
                    k = max(1, min(k, int(mask_index[j].sum().item())))
                    selected_token_indices = order[j, :k]
                    transfer_index[j, selected_token_indices] = True
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")
            
            # Transfer selected tokens
            cur_x[transfer_index] = x0[transfer_index]
        
        # Update x with decoded window
        x[:, window_start:window_end] = cur_x
        
        # Commit first mini-block of window to KV cache
        commit_start = window_start
        commit_end = window_start + mini_block_length
        commit_x = x[:, commit_start:commit_end]
        commit_position_ids = position_ids[:, commit_start:commit_end]
        
        # Build attention mask for commit: sees all previously committed + itself bidirectionally
        commit_attn_length = committed_length + mini_block_length
        commit_attn_mask = torch.ones(1, mini_block_length, commit_attn_length, device=device)
        
        model(commit_x,
              attention_mask=commit_attn_mask,
              position_ids=commit_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True)
        
        committed_mini_blocks += 1
        
        # Store confidences and tokens for the second mini-block (will be revision zone next)
        revision_zone_confidences = final_confidences[:, mini_block_length:].clone()
        revision_zone_tokens = cur_x[:, mini_block_length:].clone()
        
        # Slide window forward by one mini-block
        current_mini_block += 1
        window_idx += 1
        
        # Check stopping criteria
        if stopping_criteria_idx is not None:
            if any(stop_idx in x[:, prompt_length:window_end] for stop_idx in stopping_criteria_idx):
                break
    
    # Commit any remaining tokens (last mini-block if not committed)
    remaining_start = committed_mini_blocks * mini_block_length
    remaining_end = min(current_mini_block * mini_block_length + mini_block_length, total_length)
    
    if remaining_start < remaining_end and remaining_start < total_length:
        remaining_x = x[:, remaining_start:remaining_end]
        remaining_position_ids = position_ids[:, remaining_start:remaining_end]
        remaining_length = remaining_end - remaining_start
        committed_length = committed_mini_blocks * mini_block_length
        remaining_attn_mask = torch.ones(1, remaining_length, committed_length + remaining_length, device=device)
        
        model(remaining_x,
              attention_mask=remaining_attn_mask,
              position_ids=remaining_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True)

    return x


@torch.no_grad()
def generate_with_sdar_verification(
        model,
        sdar_model,
        prompt,
        mask_id,
        gen_length=128,
        block_length=4,
        denoising_steps=4,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        confidence_threshold=0.95,
        stopping_criteria_idx=None,
        remask_threshold=0.0,
        use_sdar_replacement=False,
        enable_sdar_verification=True,
        device=None
    ):
    """
    Generate with Block Causal mask and optional SDAR verification.
    
    This function generates text using block-causal diffusion, and optionally
    uses a separate SDAR model (in AR mode) to verify and correct generated tokens.
    
    Args:
        model: The block-causal denoising model
        sdar_model: The SDAR model for AR verification (can be None if enable_sdar_verification=False)
        prompt: Tokenized prompt dict with 'input_ids'
        mask_id: Mask token ID
        gen_length: Number of tokens to generate
        block_length: Block size for generation
        denoising_steps: Number of denoising steps per block
        temperature: Sampling temperature
        top_k: Top-k sampling (0 = disabled)
        top_p: Top-p sampling (1.0 = disabled)
        confidence_threshold: Confidence threshold for early unmask
        stopping_criteria_idx: List of stop token IDs
        remask_threshold: Only remask if sdar_conf - denoise_conf > threshold
        use_sdar_replacement: If True, directly replace with SDAR's top-1; 
                              If False, remask and regenerate
        enable_sdar_verification: If True, perform SDAR verification after each block;
                                  If False, generate normally without verification
        device: Device to use for tensor operations
    
    Returns:
        Generated token IDs tensor
    """
    model.eval()
    if sdar_model is not None:
        sdar_model.eval()
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    
    # Determine device
    if device is None:
        device = input_ids.device
        if device.type == 'cpu':
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Block causal mask
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    block_causal_mask = block_mask.repeat_interleave(block_length, dim=0).repeat_interleave(block_length, dim=1).unsqueeze(0)
    
    # AR causal mask for SDAR verification
    ar_causal_mask = torch.tril(torch.ones(total_length, total_length, device=device)).unsqueeze(0)
    
    position_ids = torch.arange(total_length, device=device).unsqueeze(0)
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids.to(device)
    
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length
    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps).to(device)

    past_key_values = DynamicCache()
    if prefill_length > 0:
        model(x[:, :prefill_length], attention_mask=block_causal_mask[:, :prefill_length, :prefill_length],
              position_ids=position_ids[:, :prefill_length], past_key_values=past_key_values, use_cache=True, store_kv=True)

    for num_block in range(prefill_blocks, num_blocks):
        block_start = num_block * block_length
        block_end = (num_block + 1) * block_length
        cur_x = x[:, block_start:block_end].clone()
        cur_attn_mask = block_causal_mask[:, block_start:block_end, :block_end]
        cur_position_ids = position_ids[:, block_start:block_end]
        
        # Track denoising probabilities
        denoise_probs = torch.zeros(block_length, device=device, dtype=torch.float32)
        
        # Denoising loop
        for step in range(denoising_steps):
            mask_index = (cur_x == mask_id)
            mask_count = int(mask_index.sum().item())
            if mask_count == 0:
                break
            outputs = model(cur_x, attention_mask=cur_attn_mask, position_ids=cur_position_ids,
                          past_key_values=past_key_values, use_cache=True, store_kv=False)
            x0, x0_p = sample_with_temperature_topk_topp(outputs.logits, temperature, top_k, top_p)
            
            confidence = torch.where(mask_index, x0_p, -torch.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            high_conf_mask = confidence[0] > confidence_threshold
            num_to_transfer = min(int(num_transfer_tokens[step].item()), mask_count)
            if num_to_transfer > 0:
                if int(high_conf_mask.sum().item()) >= num_to_transfer:
                    transfer_index[0] = high_conf_mask
                else:
                    _, idx = torch.topk(confidence[0], num_to_transfer)
                    transfer_index[0, idx] = True
            
            denoise_probs[transfer_index[0]] = x0_p[transfer_index].float()
            cur_x[transfer_index] = x0[transfer_index]

        x[:, block_start:block_end] = cur_x
        
        # SDAR Verification (optional)
        if enable_sdar_verification and sdar_model is not None:
            full_x = x[:, :block_end]
            full_pos = position_ids[:, :block_end]
            logits_sdar = sdar_model(full_x, attention_mask=block_causal_mask[:, :block_end, :block_end], position_ids=full_pos).logits[0]
            
            # Check each position for potential correction
            remask_positions = []
            sdar_replacements = {}
            for pos in range(block_length):
                gpos = block_start + pos
                if gpos < prompt_length or gpos == 0:
                    continue
                    
                actual = cur_x[0, pos].item()
                denoise_conf = denoise_probs[pos].item()
                
                probs_sdar = F.softmax(logits_sdar[gpos - 1], dim=-1)
                sdar_top1_id = probs_sdar.argmax().item()
                sdar_top1_conf = probs_sdar[sdar_top1_id].item()
                
                if sdar_top1_id != actual and sdar_top1_conf > denoise_conf + remask_threshold:
                    remask_positions.append(pos)
                    sdar_replacements[pos] = sdar_top1_id
            
            # Apply correction if needed
            if remask_positions:
                if use_sdar_replacement:
                    # Direct replacement
                    for pos in remask_positions:
                        cur_x[0, pos] = sdar_replacements[pos]
                else:
                    # Remask and regenerate
                    for pos in remask_positions:
                        cur_x[0, pos] = mask_id
                    
                    remask_steps = max(1, denoising_steps // 2)
                    remask_transfer = get_num_transfer_tokens(len(remask_positions), remask_steps)
                    
                    for step in range(remask_steps + 1):
                        mask_index = (cur_x == mask_id)
                        if mask_index.sum() == 0:
                            break
                        outputs = model(cur_x, attention_mask=cur_attn_mask, position_ids=cur_position_ids,
                                      past_key_values=past_key_values, use_cache=True, store_kv=False)
                        x0, x0_p = sample_with_temperature_topk_topp(outputs.logits, temperature, top_k, top_p)
                        
                        confidence = torch.where(mask_index, x0_p, -torch.inf)
                        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                        num_to_transfer = min(remask_transfer[step].item() if step < len(remask_transfer) else mask_index.sum().item(), 
                                             mask_index.sum().item())
                        if num_to_transfer > 0:
                            _, idx = torch.topk(confidence[0], num_to_transfer)
                            transfer_index[0, idx] = True
                        cur_x[transfer_index] = x0[transfer_index]
                
                x[:, block_start:block_end] = cur_x
        
        # Store KV for this block
        model(cur_x, attention_mask=cur_attn_mask, position_ids=cur_position_ids,
              past_key_values=past_key_values, use_cache=True, store_kv=True)
        
        # Check stopping criteria
        if stopping_criteria_idx:
            for stop_id in stopping_criteria_idx:
                if stop_id in x[0, prompt_length:block_end]:
                    return x

    return x


@torch.no_grad()
def generate_with_multi_block_verification(
        model,
        prompt,
        mask_id,
        gen_length=128,
        block_length=4,
        denoising_steps=4,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        confidence_threshold=0.85,
        stopping_criteria_idx=None,
        verify_prev_blocks=1,
        remask_threshold=0.0,
        use_direct_replacement=False,
        device=None
    ):
    """
    Generate with multi-block verification after each block completion.
    
    This function generates text block by block. After completing each block, it performs
    a forward pass to verify the current block and optionally previous X blocks. Tokens
    where the model's top-1 prediction confidence exceeds the original generation confidence
    are either remasked or directly replaced.
    
    Args:
        model: The block-causal denoising model
        prompt: Tokenized prompt dict with 'input_ids'
        mask_id: Mask token ID
        gen_length: Number of tokens to generate
        block_length: Block size for generation
        denoising_steps: Number of denoising steps per block
        temperature: Sampling temperature
        top_k: Top-k sampling (0 = disabled)
        top_p: Top-p sampling (1.0 = disabled)
        confidence_threshold: Confidence threshold for early unmask
        stopping_criteria_idx: List of stop token IDs
        verify_prev_blocks: Number of previous blocks to verify (0 = current only, 
                           1 = current + 1 previous, etc.)
        remask_threshold: Only remask/replace if model_conf - orig_conf > threshold
        use_direct_replacement: If True, directly replace with model's top-1;
                               If False, remask and regenerate with expanded block
        device: Device to use for tensor operations
    
    Returns:
        Generated token IDs tensor
    """
    model.eval()
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    
    # Determine device
    if device is None:
        device = input_ids.device
        if device.type == 'cpu':
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Block causal mask
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    block_causal_mask = block_mask.repeat_interleave(block_length, dim=0).repeat_interleave(block_length, dim=1).unsqueeze(0)
    
    position_ids = torch.arange(total_length, device=device).unsqueeze(0)
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids.to(device)
    
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length
    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)

    past_key_values = DynamicCache()
    if prefill_length > 0:
        model(x[:, :prefill_length], attention_mask=block_causal_mask[:, :prefill_length, :prefill_length],
              position_ids=position_ids[:, :prefill_length], past_key_values=past_key_values, use_cache=True, store_kv=True)

    # Track original generation confidences for each block
    block_confidences = {}  # block_idx -> tensor of shape [block_length]
    
    for num_block in range(prefill_blocks, num_blocks):
        block_start = num_block * block_length
        block_end = (num_block + 1) * block_length
        cur_x = x[:, block_start:block_end].clone()
        cur_attn_mask = block_causal_mask[:, block_start:block_end, :block_end]
        cur_position_ids = position_ids[:, block_start:block_end]
        
        # Track denoising probabilities for this block
        denoise_probs = torch.zeros(block_length, device=device, dtype=torch.float32)
        
        # Denoising loop
        for step in range(denoising_steps):
            mask_index = (cur_x == mask_id)
            mask_count = int(mask_index.sum().item())
            if mask_count == 0:
                break
            outputs = model(cur_x, attention_mask=cur_attn_mask, position_ids=cur_position_ids,
                          past_key_values=past_key_values, use_cache=True, store_kv=False)
            x0, x0_p = sample_with_temperature_topk_topp(outputs.logits, temperature, top_k, top_p)
            
            confidence = torch.where(mask_index, x0_p, -torch.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            high_conf_mask = confidence[0] > confidence_threshold
            num_to_transfer = min(int(num_transfer_tokens[step].item()), mask_count)
            if num_to_transfer > 0:
                if int(high_conf_mask.sum().item()) >= num_to_transfer:
                    transfer_index[0] = high_conf_mask
                else:
                    _, idx = torch.topk(confidence[0], num_to_transfer)
                    transfer_index[0, idx] = True
            
            denoise_probs[transfer_index[0]] = x0_p[transfer_index].float()
            cur_x[transfer_index] = x0[transfer_index]

        x[:, block_start:block_end] = cur_x
        block_confidences[num_block] = denoise_probs.clone()
        
        # Multi-block verification
        # Determine which blocks to verify
        verify_start_block = max(prefill_blocks, num_block - verify_prev_blocks)
        verify_start_pos = verify_start_block * block_length
        verify_end_pos = block_end
        verify_length = verify_end_pos - verify_start_pos
        num_verify_blocks = num_block - verify_start_block + 1
        
        # Forward pass for verification (use full prefix to avoid mask/len mismatch)
        full_x = x[:, :verify_end_pos]
        full_attn_mask = block_causal_mask[:, :verify_end_pos, :verify_end_pos]
        full_position_ids = position_ids[:, :verify_end_pos]
        
        # Get verification logits for the full prefix
        verify_logits = model(full_x, attention_mask=full_attn_mask, position_ids=full_position_ids,
                             use_cache=False, store_kv=False).logits[0]
        
        # Check each position for potential correction
        # For block-causal model, we verify within each block using bidirectional context
        remask_positions = []  # list of (global_pos, local_pos_in_verify_window)
        replacement_tokens = {}  # global_pos -> token_id
        
        for blk_idx in range(verify_start_block, num_block + 1):
            blk_start = blk_idx * block_length
            blk_end = (blk_idx + 1) * block_length
            
            for pos_in_block in range(block_length):
                gpos = blk_start + pos_in_block
                local_pos = gpos - verify_start_pos  # position in verify window
                
                if gpos < prompt_length or gpos == 0:
                    continue
                    
                actual_token = x[0, gpos].item()
                orig_conf = block_confidences[blk_idx][pos_in_block].item()
                
                # Get model's prediction for token at gpos using AR-style logits
                probs = F.softmax(verify_logits[gpos - 1], dim=-1)
                top1_id = probs.argmax().item()
                top1_conf = probs[top1_id].item()
                
                # Check if model disagrees with higher confidence
                if top1_id != actual_token and top1_conf > orig_conf + remask_threshold:
                    remask_positions.append((gpos, local_pos))
                    replacement_tokens[gpos] = top1_id
        
        # Apply correction if needed
        if remask_positions:
            if use_direct_replacement:
                # Direct replacement with model's top-1
                for gpos, _ in remask_positions:
                    x[0, gpos] = replacement_tokens[gpos]
                    # Update confidence
                    blk_idx = gpos // block_length
                    pos_in_block = gpos % block_length
                    block_confidences[blk_idx][pos_in_block] = 1.0  # Assume high confidence after replacement
            else:
                # Remask and regenerate with expanded block size
                # expanded_block_size = (verify_prev_blocks + 1) * block_length
                
                # Remask the identified positions
                for gpos, _ in remask_positions:
                    x[0, gpos] = mask_id
                
                # Regenerate using the verification window as a larger block
                # Number of denoising steps proportional to remasked positions
                num_remasked = len(remask_positions)
                remask_steps = max(1, min(denoising_steps, num_remasked))
                remask_transfer = get_num_transfer_tokens(num_remasked, remask_steps)
                
                regen_x = x[:, verify_start_pos:verify_end_pos].clone()
                regen_attn_mask = block_causal_mask[:, verify_start_pos:verify_end_pos, :verify_end_pos]
                regen_position_ids = position_ids[:, verify_start_pos:verify_end_pos]
                
                # Build a fresh cache aligned to verify_start_pos (match notebook behavior).
                regen_past_key_values = DynamicCache()
                if verify_start_pos > 0:
                    model(x[:, :verify_start_pos],
                          attention_mask=block_causal_mask[:, :verify_start_pos, :verify_start_pos],
                          position_ids=position_ids[:, :verify_start_pos],
                          past_key_values=regen_past_key_values,
                          use_cache=True,
                          store_kv=True)
                
                for step in range(remask_steps + 1):
                    mask_index = (regen_x == mask_id)
                    if mask_index.sum() == 0:
                        break
                    
                    outputs = model(regen_x, attention_mask=regen_attn_mask, position_ids=regen_position_ids,
                                  past_key_values=regen_past_key_values, use_cache=True, store_kv=False)
                    x0, x0_p = sample_with_temperature_topk_topp(outputs.logits, temperature, top_k, top_p)
                    
                    confidence = torch.where(mask_index, x0_p, -torch.inf)
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    num_to_transfer = min(remask_transfer[step].item() if step < len(remask_transfer) else mask_index.sum().item(),
                                         mask_index.sum().item())
                    if num_to_transfer > 0:
                        _, idx = torch.topk(confidence[0], num_to_transfer)
                        transfer_index[0, idx] = True
                    
                    # Update confidences for regenerated tokens
                    for i in range(verify_length):
                        if transfer_index[0, i]:
                            gpos = verify_start_pos + i
                            blk_idx = gpos // block_length
                            pos_in_block = gpos % block_length
                            if blk_idx in block_confidences:
                                block_confidences[blk_idx][pos_in_block] = x0_p[0, i].item()
                    
                    regen_x[transfer_index] = x0[transfer_index]
                
                x[:, verify_start_pos:verify_end_pos] = regen_x
        
        # Store KV for this block
        model(x[:, block_start:block_end], attention_mask=cur_attn_mask, position_ids=cur_position_ids,
              past_key_values=past_key_values, use_cache=True, store_kv=True)
        
        # Check stopping criteria
        if stopping_criteria_idx:
            for stop_id in stopping_criteria_idx:
                if stop_id in x[0, prompt_length:block_end]:
                    return x

    return x


# ============================================================================
# Begin revision by Yifan Yu: block diffusion generation with Dream-style
# token shift. During training with the shift, hidden[i] predicts token[i+1].
# At inference, we apply the same shift to logits so that logits[i] gives
# the prediction for the token that should go at position i.
# Reference: Dream 7B (arxiv 2508.15487), Fast-dLLM (NVlabs)
# ============================================================================
@torch.no_grad()
def block_diffusion_generate_with_shift(
        model,
        prompt,
        mask_id,
        gen_length=128,
        block_length=8,
        denoising_steps=8,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy='low_confidence_dynamic',
        confidence_threshold=0.85,
        eb_threshold=None,
        stopping_criteria_idx=None,
        causal_prefill=False,
        causal_commit=False
    ):
    """
    Block diffusion generation with Dream-style logit shift.

    With Dream shift training, hidden[i] predicts token[i+1]. At inference we
    need logits[i] to predict the token at position i. We achieve this by:
    1. Caching the last-position logits from the previous block (or prefill).
       This predicts the first token of the current block.
    2. Shifting: logits = torch.cat([prev_last_logits, logits[:, :-1]], dim=1)
       so logits[0] = prev block's last prediction (for current pos 0),
       logits[1] = current hidden[0]'s prediction (for pos 1), etc.

    Args:
        causal_prefill: If True, use regular causal attention for the prompt
            prefill (matching training where clean prefix uses causal attention).
            If False, use block-causal attention for prefill (legacy behavior).
        causal_commit: If True, use causal attention (not block-causal) for the
            store_kv commit pass after denoising. This means each token in the
            committed block only sees previous tokens in its KV, rather than
            all tokens in the block bidirectionally. Modified by Yifan Yu.
    """

    model.eval()
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length +
                  block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(
        num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                               .repeat_interleave(block_length, dim=1).unsqueeze(0)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    # Prefill stage — also capture last-position logits for the first decode block
    prev_block_last_logits = None
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        if causal_prefill:
            # Regular causal attention for prompt (matches training where
            # clean prefix uses standard causal, not block-causal)
            cur_attn_mask = torch.tril(torch.ones(
                prefill_length, prefill_length, device=model.device)).unsqueeze(0)
        else:
            cur_attn_mask = block_diffusion_attention_mask[:,
                                                           :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        prefill_output = model(cur_x,
              attention_mask=cur_attn_mask,
              position_ids=cur_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True)
        # Last position's logits predict the first token of the next block
        prev_block_last_logits = prefill_output.logits[:, -1:, :]  # (1, 1, vocab)

    num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)

    # Decode stage
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
        ]
        cur_position_ids = position_ids[:, num_block *
                                        block_length:(num_block+1)*block_length]

        # Build causal commit mask: same as cur_attn_mask but with causal
        # (lower-triangular) within the current block instead of bidirectional.
        # Only used when causal_commit=True for the store_kv pass.
        if causal_commit:
            commit_attn_mask = cur_attn_mask.clone()
            # The last block_length columns correspond to the current block.
            # Replace them with lower-triangular (causal within block).
            causal_block = torch.tril(torch.ones(
                block_length, block_length, device=model.device))
            commit_attn_mask[:, :, -block_length:] = causal_block
        else:
            commit_attn_mask = cur_attn_mask

        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                # Store kv cache and capture last-position logits for next block
                store_output = model(cur_x,
                      attention_mask=commit_attn_mask,
                      position_ids=cur_position_ids,
                      past_key_values=past_key_values,
                      use_cache=True,
                      store_kv=True)
                prev_block_last_logits = store_output.logits[:, -1:, :]
                break

            # Denoising
            logits = model(cur_x,
                           attention_mask=cur_attn_mask,
                           position_ids=cur_position_ids,
                           past_key_values=past_key_values,
                           use_cache=True,
                           store_kv=False).logits

            # Dream-style logit shift with proper first-position handling:
            # prev_block_last_logits predicts current block's pos 0
            # logits[:, 0] predicts pos 1, logits[:, 1] predicts pos 2, etc.
            if prev_block_last_logits is not None:
                logits = torch.cat([prev_block_last_logits, logits[:, :-1]], dim=1)
            else:
                # Fallback for first block with no prefill (shouldn't normally happen)
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            # Sampling
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # Sampling strategy
            if remasking_strategy == 'sequential':
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(cur_x.shape[0]):
                    if mask_index[j].any():
                        first_mask_index = mask_index[j].nonzero(as_tuple=True)[
                            0].min().item()
                        transfer_index[j, first_mask_index:first_mask_index +
                                       num_transfer_tokens[step]] = True
                    else:
                        raise ValueError(
                            "No mask tokens found in the current block.")

            elif remasking_strategy == 'low_confidence_static':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, idx = torch.topk(
                        confidence[j], num_transfer_tokens[step])
                    transfer_index[j, idx] = True

            elif remasking_strategy == 'low_confidence_dynamic':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    high_conf_mask = confidence[j] > confidence_threshold
                    num_high_confidence = high_conf_mask.sum()
                    if num_high_confidence >= num_transfer_tokens[step]:
                        transfer_index[j] = high_conf_mask
                    else:
                        _, idx = torch.topk(
                            confidence[j], num_transfer_tokens[step])
                        transfer_index[j, idx] = True
            elif remasking_strategy == 'low_confidence_causal':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(cur_x.shape[0]):
                    if not mask_index[j].any():
                        continue
                    high_conf_mask = confidence[j] > confidence_threshold
                    if high_conf_mask.sum() >= num_transfer_tokens[step]:
                        transfer_index[j] = high_conf_mask
                    else:
                        first_mask = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                        transfer_index[j, first_mask] = True
            elif remasking_strategy == 'low_confidence_causal2':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(cur_x.shape[0]):
                    if not mask_index[j].any():
                        continue
                    # Always commit leftmost masked token
                    first_mask = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                    transfer_index[j, first_mask] = True
                    # Plus all tokens exceeding confidence threshold
                    high_conf_mask = confidence[j] > confidence_threshold
                    transfer_index[j] = transfer_index[j] | high_conf_mask
            elif remasking_strategy == "entropy_bounded":
                eps = 1e-12
                entropies = -(x0_p.clamp_min(eps) * (x0_p.clamp_min(eps)).log()).sum(dim=-1)
                entropies = torch.where(mask_index, entropies, torch.inf)
                ent_sorted, order = torch.sort(entropies, dim=1, descending=False)
                cumsum = torch.cumsum(ent_sorted, dim=1)
                for j in range(x0_p.shape[0]):
                    k = torch.searchsorted(cumsum[j], torch.tensor(eb_threshold, device=x0_p.device), right=False).item()
                    k = max(1, min(k, int(mask_index[j].sum().item())))
                    selected_token_indices = order[j, :k]
                    transfer_index[j, selected_token_indices] = True

            else:
                raise ValueError(
                    f"Unknown remasking strategy: {remasking_strategy}")

            cur_x[transfer_index] = x0[transfer_index]
        else:
            # for-else: executes if loop completed without break
            # Store KV and capture last-position logits for next block
            store_output = model(cur_x,
                  attention_mask=commit_attn_mask,
                  position_ids=cur_position_ids,
                  past_key_values=past_key_values,
                  use_cache=True,
                  store_kv=True)
            prev_block_last_logits = store_output.logits[:, -1:, :]

        x[:, num_block*block_length:(num_block+1)*block_length] = cur_x
        if stopping_criteria_idx is not None and any(stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx):
            break

    return x
# End revision by Yifan Yu
# ============================================================================


# ============================================================================
# Block diffusion generation with Dream-style logit shift + self-verification
# Modified by Yifan Yu
#
# After each block is denoised and committed, verifies the generated tokens
# by checking the commit logits. If the model's shifted logit at position i
# assigns low probability to the actual token at position i+1, positions
# from the disagreement onward are remasked and re-denoised.
#
# Example:
#   Generated block: [t1, t2, t3, t4]
#   Commit logits (shifted): hidden[t1]->pred[t2], hidden[t2]->pred[t3], ...
#   If hidden[t2] gives low prob on t3:
#     remask from t3 onward: [t1, t2, MASK, MASK]
#     re-denoise to fill MASKs
#
# Reference: Dream 7B (arxiv 2508.15487), Fast-dLLM (NVlabs)
# ============================================================================
@torch.no_grad()
def block_diffusion_generate_with_shift_verified(
        model,
        prompt,
        mask_id,
        gen_length=128,
        block_length=8,
        denoising_steps=8,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy='low_confidence_dynamic',
        confidence_threshold=0.85,
        eb_threshold=None,
        stopping_criteria_idx=None,
        causal_prefill=False,
        causal_commit=False,
        # Verification parameters
        verify_threshold=0.3,
        verify_mode='logprob',
        max_verify_retries=2,
    ):
    """
    Block diffusion generation with Dream-style logit shift and self-verification.

    Same as block_diffusion_generate_with_shift, but after each block's commit
    pass, checks whether the committed tokens are consistent with the model's
    shifted logits. If a token has low log-probability under the commit logits,
    it and all subsequent tokens in the block are remasked and re-denoised.

    Args:
        verify_threshold: Threshold for verification. Interpretation depends
            on verify_mode:
            - 'logprob': remask if log_prob(token) < log(verify_threshold)
                         i.e., if P(token) < verify_threshold
            - 'rank': remask if token is not in top-k predictions (threshold=k)
            - 'disagreement': remask if argmax prediction != actual token
        verify_mode: How to detect bad tokens:
            - 'logprob': check if probability < verify_threshold
            - 'rank': check if token is not in top verify_threshold predictions
            - 'disagreement': check if model's top-1 != actual token
        max_verify_retries: Max number of re-denoising attempts per block.
        Other args same as block_diffusion_generate_with_shift.

    Returns:
        Generated token IDs tensor
    """

    model.eval()
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length +
                  block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(
        num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                               .repeat_interleave(block_length, dim=1).unsqueeze(0)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    # Prefill
    prev_block_last_logits = None
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        if causal_prefill:
            cur_attn_mask = torch.tril(torch.ones(
                prefill_length, prefill_length, device=model.device)).unsqueeze(0)
        else:
            cur_attn_mask = block_diffusion_attention_mask[:,
                                                           :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        prefill_output = model(cur_x,
              attention_mask=cur_attn_mask,
              position_ids=cur_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True)
        prev_block_last_logits = prefill_output.logits[:, -1:, :]

    original_num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)

    # Decode
    for num_block in range(prefill_blocks, num_blocks):
        block_start = num_block * block_length
        block_end = (num_block + 1) * block_length
        cur_x = x[:, block_start:block_end].clone()
        # Reset transfer tokens for this block (may have been modified by retry)
        num_transfer_tokens = original_num_transfer_tokens.clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, block_start:block_end, :block_end
        ]
        cur_position_ids = position_ids[:, num_block *
                                        block_length:(num_block+1)*block_length]

        # Build commit mask (causal within block if causal_commit)
        if causal_commit:
            commit_attn_mask = cur_attn_mask.clone()
            causal_block = torch.tril(torch.ones(
                block_length, block_length, device=model.device))
            commit_attn_mask[:, :, -block_length:] = causal_block
        else:
            commit_attn_mask = cur_attn_mask

        # Save the original prev_block_last_logits from the previous block
        # so we can restore it on retries (commit overwrites it)
        original_prev_block_last_logits = prev_block_last_logits.clone() if prev_block_last_logits is not None else None

        for verify_attempt in range(max_verify_retries + 1):
            # --- Denoising loop ---
            for step in range(denoising_steps + 1):
                mask_index = (cur_x == mask_id)
                if mask_index.sum() == 0:
                    break

                logits = model(cur_x,
                               attention_mask=cur_attn_mask,
                               position_ids=cur_position_ids,
                               past_key_values=past_key_values,
                               use_cache=True,
                               store_kv=False).logits

                # Dream-style logit shift
                if prev_block_last_logits is not None:
                    logits = torch.cat([prev_block_last_logits, logits[:, :-1]], dim=1)
                else:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Sampling
                x0, x0_p = sample_with_temperature_topk_topp(
                    logits, temperature=temperature, top_k=top_k, top_p=top_p)

                # Remasking strategy
                # On retry (verify_attempt > 0), use left-to-right (always leftmost only)
                effective_strategy = 'left_to_right' if verify_attempt > 0 else remasking_strategy

                if effective_strategy == 'left_to_right':
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(cur_x.shape[0]):
                        if not mask_index[j].any():
                            continue
                        first_mask = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                        transfer_index[j, first_mask] = True

                elif effective_strategy == 'sequential':
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(cur_x.shape[0]):
                        if mask_index[j].any():
                            first_mask_index = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                            transfer_index[j, first_mask_index:first_mask_index +
                                           num_transfer_tokens[step]] = True

                elif effective_strategy == 'low_confidence_static':
                    confidence = torch.where(mask_index, x0_p, -torch.inf)
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(confidence.shape[0]):
                        _, idx = torch.topk(confidence[j], num_transfer_tokens[step])
                        transfer_index[j, idx] = True

                elif effective_strategy == 'low_confidence_dynamic':
                    confidence = torch.where(mask_index, x0_p, -torch.inf)
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(confidence.shape[0]):
                        high_conf_mask = confidence[j] > confidence_threshold
                        if high_conf_mask.sum() >= num_transfer_tokens[step]:
                            transfer_index[j] = high_conf_mask
                        else:
                            _, idx = torch.topk(confidence[j], num_transfer_tokens[step])
                            transfer_index[j, idx] = True

                elif effective_strategy == 'low_confidence_causal':
                    confidence = torch.where(mask_index, x0_p, -torch.inf)
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(cur_x.shape[0]):
                        if not mask_index[j].any():
                            continue
                        high_conf_mask = confidence[j] > confidence_threshold
                        if high_conf_mask.sum() >= num_transfer_tokens[step]:
                            transfer_index[j] = high_conf_mask
                        else:
                            first_mask = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                            transfer_index[j, first_mask] = True

                elif effective_strategy == 'low_confidence_causal2':
                    confidence = torch.where(mask_index, x0_p, -torch.inf)
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(cur_x.shape[0]):
                        if not mask_index[j].any():
                            continue
                        first_mask = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                        transfer_index[j, first_mask] = True
                        high_conf_mask = confidence[j] > confidence_threshold
                        transfer_index[j] = transfer_index[j] | high_conf_mask

                else:
                    raise ValueError(f"Unknown remasking strategy: {effective_strategy}")

                cur_x[transfer_index] = x0[transfer_index]

            # --- Commit pass: store KV and get logits for verification ---
            # Use the original prev_block_last_logits from the previous block
            # (not from a prior commit attempt on this block)
            prev_block_last_logits_before_commit = original_prev_block_last_logits

            store_output = model(cur_x,
                  attention_mask=commit_attn_mask,
                  position_ids=cur_position_ids,
                  past_key_values=past_key_values,
                  use_cache=True,
                  store_kv=True)
            commit_logits = store_output.logits  # (1, block_length, vocab)
            prev_block_last_logits = commit_logits[:, -1:, :]

            # --- Self-verification ---
            # With Dream shift: commit_logits[i] predicts token at position i+1
            # prev_block_last_logits_before_commit predicts cur_x[0]
            # Check positions 0..block_length-2 predicting 1..block_length-1
            if verify_attempt >= max_verify_retries:
                break  # No more retries, accept as is

            # Only verify generated tokens (skip prompt overflow)
            gen_start_in_block = max(0, prompt_length - block_start)

            # Compute probabilities of actual tokens under shifted logits
            # With Dream shift: logits[i] predicts token[i+1]
            # Additionally, prev_block_last_logits predicts the first token
            # in this block (cur_x[0]).
            remask_from = None
            probs = F.softmax(commit_logits[0], dim=-1)  # (block_length, vocab)

            # First: verify cur_x[gen_start_in_block] using prev_block_last_logits
            # (the previous block's last hidden state predicts this block's first token)
            if prev_block_last_logits_before_commit is not None and gen_start_in_block < block_length:
                first_gen_token = cur_x[0, gen_start_in_block].item()
                first_probs = F.softmax(prev_block_last_logits_before_commit[0, 0], dim=-1)
                first_token_prob = first_probs[first_gen_token].item()
                first_top1_id = first_probs.argmax().item()

                should_remask_first = False
                if verify_mode == 'logprob':
                    should_remask_first = (first_token_prob < verify_threshold)
                elif verify_mode == 'rank':
                    _, topk_ids = first_probs.topk(int(verify_threshold))
                    should_remask_first = (first_gen_token not in topk_ids)
                elif verify_mode == 'disagreement':
                    should_remask_first = (first_top1_id != first_gen_token)

                if should_remask_first:
                    remask_from = gen_start_in_block

            # Then: verify remaining positions using commit logits
            # commit_logits[pos] predicts cur_x[pos+1]
            if remask_from is None:
                for pos in range(gen_start_in_block, block_length - 1):
                    next_token = cur_x[0, pos + 1].item()
                    token_prob = probs[pos, next_token].item()
                    top1_id = probs[pos].argmax().item()

                    should_remask = False
                    if verify_mode == 'logprob':
                        should_remask = (token_prob < verify_threshold)
                    elif verify_mode == 'rank':
                        _, topk_ids = probs[pos].topk(int(verify_threshold))
                        should_remask = (next_token not in topk_ids)
                    elif verify_mode == 'disagreement':
                        should_remask = (top1_id != next_token)
                    else:
                        raise ValueError(f"Unknown verify_mode: {verify_mode}")

                    if should_remask:
                        remask_from = pos + 1
                        break

            if remask_from is None:
                break  # All tokens verified, accept this block

            # --- Remask from disagreement point onward ---
            # Roll back the KV cache: remove this block's KV entries
            # (we stored them in the commit pass, need to undo)
            # Compatible with multiple DynamicCache versions:
            #   - transformers <=4.52: key_cache/value_cache attributes
            #   - transformers >=4.57: layers[i].keys/values attributes
            for layer_idx in range(len(past_key_values)):
                if hasattr(past_key_values, 'key_cache'):
                    k = past_key_values.key_cache[layer_idx]
                    v = past_key_values.value_cache[layer_idx]
                    past_key_values.key_cache[layer_idx] = k[:, :, :-block_length, :]
                    past_key_values.value_cache[layer_idx] = v[:, :, :-block_length, :]
                elif hasattr(past_key_values, 'layers'):
                    layer = past_key_values.layers[layer_idx]
                    layer.keys = layer.keys[:, :, :-block_length, :]
                    layer.values = layer.values[:, :, :-block_length, :]

            # Restore prev_block_last_logits so the denoising shift is correct
            prev_block_last_logits = original_prev_block_last_logits

            # Remask: keep tokens before remask_from, mask the rest
            cur_x[0, remask_from:] = mask_id

            # Recompute num_transfer_tokens for the remaining positions
            num_remaining = block_length - remask_from
            remaining_steps = max(1, min(denoising_steps, num_remaining))
            num_transfer_tokens_retry = get_num_transfer_tokens(
                num_remaining, remaining_steps)
            # Pad to match denoising_steps+1 length
            num_transfer_tokens = torch.zeros(denoising_steps + 1, dtype=torch.int64)
            num_transfer_tokens[:remaining_steps] = num_transfer_tokens_retry

            # Loop back to denoising (verify_attempt increments)

        # Write back
        x[:, block_start:block_end] = cur_x
        if stopping_criteria_idx is not None and any(
                stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx):
            break

    return x
# End revision by Yifan Yu
# ============================================================================


# ============================================================================
# Batched block diffusion generation with Dream-style logit shift
# Modified by Yifan Yu
# Processes multiple prompts in parallel for better GPU utilization.
# Uses the same KV cache mechanism as the single-sample version:
#   - store_kv=False during denoising (read-only cache)
#   - store_kv=True after block is finalized (cache grows by block_length)
# Reference: Dream 7B (arxiv 2508.15487), Fast-dLLM (NVlabs)
# ============================================================================

@torch.no_grad()
def block_diffusion_generate_with_shift_batched(
        model,
        prompts,  # list of dicts, each with 'input_ids' tensor of shape (1, seq_len)
        mask_id,
        pad_token_id=151643,  # <|endoftext|> for Qwen3
        gen_length=128,
        block_length=8,
        denoising_steps=8,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy='low_confidence_dynamic',
        confidence_threshold=0.85,
        eb_threshold=None,
        stopping_criteria_idx=None,
        causal_prefill=False,
        causal_commit=False
    ):
    """
    Batched block diffusion generation with Dream-style logit shift.

    Processes multiple prompts in parallel for better GPU utilization.
    Uses the same KV cache mechanism as the single-sample version:
      - store_kv=False during denoising (read-only cache)
      - store_kv=True after block is finalized (cache grows by block_length)

    Prompts are padded to a common length. Blocks in the padding region
    between shorter prompts and the max prompt length are handled naturally:
    real prompt tokens have mask_index=False so they pass through denoising
    untouched, and their KV is stored when the block finalizes.

    Args:
        prompts: list of dicts, each with 'input_ids' tensor of shape (1, seq_len)
        causal_prefill: If True, use regular causal attention for the prompt
            prefill (matching training where clean prefix uses causal attention).
            If False, use block-causal attention for prefill (legacy behavior).
        Other args same as block_diffusion_generate_with_shift.

    Returns:
        x: tensor of shape (batch, total_length) with generated tokens
        prompt_lengths: list of original prompt lengths per sample
    """

    model.eval()
    batch_size = len(prompts)
    device = model.device

    # Collect prompt lengths
    prompt_lengths = [p['input_ids'].shape[1] for p in prompts]
    max_prompt_length = max(prompt_lengths)

    # Compute total sequence length (padded to multiple of block_length)
    # Prefill length: round UP so all prompt tokens stay in causal prefill.
    # At most block_length-1 extra pad tokens are added to the prefill window.
    prefill_blocks = (max_prompt_length + block_length - 1) // block_length 
    # prefill_blocks = 0
    prefill_length = prefill_blocks * block_length

    # Total blocks: prefill blocks plus generation blocks (ceil on gen_length).
    num_blocks = prefill_blocks + (gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Left-pad prompts so they end at prefill_length (causal prefill covers all real tokens).
    # Generation region (after prefill_length) filled with mask_id.
    x = torch.full((batch_size, total_length), mask_id,
                   dtype=torch.long, device=device)
    pad_offsets = [max(0, prefill_length - prompt_lengths[i])
                   for i in range(batch_size)]
    for i, p in enumerate(prompts):
        plen = prompt_lengths[i]
        start = pad_offsets[i]
        x[i, start:start + plen] = p['input_ids'][0]
        if start > 0:
            x[i, :start] = pad_token_id

    # Attention mask: block-causal, shared across batch (broadcasts over dim 0)
    block_mask = torch.tril(torch.ones(
        num_blocks, num_blocks, device=device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                               .repeat_interleave(block_length, dim=1).unsqueeze(0)
    # Per-sample position_ids: real tokens start from 0, left-padding gets 0
    position_ids = torch.zeros(batch_size, total_length, dtype=torch.long, device=device)
    for i in range(batch_size):
        position_ids[i, pad_offsets[i]:] = torch.arange(
            total_length - pad_offsets[i], device=device)

    past_key_values = DynamicCache()
    prev_block_last_logits = None

    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        if causal_prefill:
            # Per-sample causal mask excluding left-padding
            # Shape: (batch, 1, prefill_length, prefill_length) for SDPA compatibility
            causal = torch.tril(torch.ones(
                prefill_length, prefill_length, device=device))
            cur_attn_mask = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).clone()
            for i in range(batch_size):
                if pad_offsets[i] > 0:
                    cur_attn_mask[i, 0, :, :pad_offsets[i]] = 0
                    cur_attn_mask[i, 0, :pad_offsets[i], :] = 0
        # else:
        #     cur_attn_mask = block_diffusion_attention_mask[:,
        #                                                    :prefill_length, :prefill_length]
        else:
            base = block_diffusion_attention_mask[:, :prefill_length, :prefill_length]
            cur_attn_mask = base.unsqueeze(0).expand(batch_size, 1, -1, -1).clone()
            for i in range(batch_size):
                if pad_offsets[i] > 0:
                    cur_attn_mask[i, 0, :, :pad_offsets[i]] = 0
                    cur_attn_mask[i, 0, :pad_offsets[i], :] = 0
        cur_position_ids = position_ids[:, :prefill_length]
        prefill_output = model(cur_x,
              attention_mask=cur_attn_mask,
              position_ids=cur_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True)
        # All prompts end at prefill_length, so last position works for all
        prev_block_last_logits = prefill_output.logits[:, -1:, :]  # (batch, 1, vocab)

    num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)

    # Per-sample stopping
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Precompute per-sample mask to exclude left-padding from prefill KV
    # Shape: (batch, 1, 1, prefill_length) for 4D SDPA compatibility
    prefill_valid_mask = torch.ones(batch_size, 1, 1, prefill_length, device=device)
    for i in range(batch_size):
        if pad_offsets[i] > 0:
            prefill_valid_mask[i, 0, 0, :pad_offsets[i]] = 0

    # Decode stage
    for num_block in range(prefill_blocks, num_blocks):
        if finished.all():
            break

        block_start = num_block * block_length
        block_end = (num_block + 1) * block_length
        cur_x = x[:, block_start:block_end].clone()
        # Per-sample decode mask: block-causal + exclude left-padding from prefill
        # Shape: (batch, 1, block_length, block_end) for 4D SDPA compatibility
        base_mask = block_diffusion_attention_mask[
            :, block_start:block_end, :block_end
        ].unsqueeze(0)  # (1, 1, block_length, block_end)
        cur_attn_mask = base_mask.expand(batch_size, 1, -1, -1).clone()
        cur_attn_mask[:, :, :, :prefill_length] *= prefill_valid_mask
        cur_position_ids = position_ids[:, block_start:block_end]

        # Build causal commit mask: same as cur_attn_mask but with causal
        # (lower-triangular) within the current block instead of bidirectional.
        # Only used when causal_commit=True for the store_kv pass.
        # Modified by Yifan Yu
        if causal_commit:
            commit_attn_mask = cur_attn_mask.clone()
            # Last block_length columns = current block. Replace with causal.
            causal_block = torch.tril(torch.ones(
                block_length, block_length, device=device))
            commit_attn_mask[:, :, :, -block_length:] = causal_block
        else:
            commit_attn_mask = cur_attn_mask

        # For finished samples, replace masks with a non-mask token
        # so denoising is a no-op for them
        if finished.any():
            fill_token = stopping_criteria_idx[0] if stopping_criteria_idx else 0
            for i in range(batch_size):
                if finished[i]:
                    cur_x[i] = fill_token

        # Denoising steps
        for step in range(denoising_steps):
            mask_index = (cur_x == mask_id)
            if not mask_index.any():
                break  # Entire batch has no masks in this block

            logits = model(cur_x,
                           attention_mask=cur_attn_mask,
                           position_ids=cur_position_ids,
                           past_key_values=past_key_values,
                           use_cache=True,
                           store_kv=False).logits

            # Dream-style logit shift
            if prev_block_last_logits is not None:
                logits = torch.cat([prev_block_last_logits, logits[:, :-1]], dim=1)
            else:
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            # Sampling
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # Remasking strategy
            if remasking_strategy == 'sequential':
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(batch_size):
                    if not mask_index[j].any():
                        continue
                    first_mask = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                    num_masked = mask_index[j].sum().item()
                    k = min(num_transfer_tokens[step].item(), num_masked)
                    transfer_index[j, first_mask:first_mask + k] = True

            elif remasking_strategy == 'low_confidence_static':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(batch_size):
                    if not mask_index[j].any():
                        continue
                    num_masked = mask_index[j].sum().item()
                    k = min(num_transfer_tokens[step].item(), num_masked)
                    _, idx = torch.topk(confidence[j], k)
                    transfer_index[j, idx] = True

            elif remasking_strategy == 'low_confidence_dynamic':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(batch_size):
                    if not mask_index[j].any():
                        continue
                    high_conf_mask = confidence[j] > confidence_threshold
                    num_high_confidence = high_conf_mask.sum()
                    if num_high_confidence >= num_transfer_tokens[step]:
                        transfer_index[j] = high_conf_mask
                    else:
                        num_masked = mask_index[j].sum().item()
                        k = min(num_transfer_tokens[step].item(), num_masked)
                        _, idx = torch.topk(confidence[j], k)
                        transfer_index[j, idx] = True

            elif remasking_strategy == 'low_confidence_causal':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(batch_size):
                    if not mask_index[j].any():
                        continue
                    high_conf_mask = confidence[j] > confidence_threshold
                    if high_conf_mask.sum() >= num_transfer_tokens[step]:
                        transfer_index[j] = high_conf_mask
                    else:
                        first_mask = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                        transfer_index[j, first_mask] = True

            elif remasking_strategy == 'low_confidence_causal2':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(batch_size):
                    if not mask_index[j].any():
                        continue
                    first_mask = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                    transfer_index[j, first_mask] = True
                    high_conf_mask = confidence[j] > confidence_threshold
                    transfer_index[j] = transfer_index[j] | high_conf_mask

            else:
                raise ValueError(
                    f"Unknown remasking strategy: {remasking_strategy}")

            # Only transfer at masked positions (safety guard for batching)
            transfer_index = transfer_index & mask_index
            cur_x[transfer_index] = x0[transfer_index]

        # Always store KV after denoising (cache grows by block_length)
        # Use commit_attn_mask which is causal within block if causal_commit=True
        store_output = model(cur_x,
              attention_mask=commit_attn_mask,
              position_ids=cur_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True)
        prev_block_last_logits = store_output.logits[:, -1:, :]

        # Write back denoised block
        x[:, block_start:block_end] = cur_x

        # Check EOS per sample (generation starts after each sample's prompt)
        if stopping_criteria_idx is not None:
            for i in range(batch_size):
                if not finished[i]:
                    gen_start = pad_offsets[i] + prompt_lengths[i]
                    gen_region = x[i, gen_start:]
                    if any(stop_idx in gen_region for stop_idx in stopping_criteria_idx):
                        finished[i] = True

    # Return x and per-sample generation start positions
    gen_start_positions = [pad_offsets[i] + prompt_lengths[i] for i in range(batch_size)]
    return x, gen_start_positions









# @torch.no_grad()
# def block_diffusion_generate_with_shift_batched(
#         model,
#         prompts,  # list of dicts, each with 'input_ids' tensor of shape (1, seq_len)
#         mask_id,
#         pad_token_id=151643,  # <|endoftext|> for Qwen3
#         gen_length=128,
#         block_length=8,
#         denoising_steps=8,
#         temperature=1.0,
#         top_k=0,
#         top_p=1.0,
#         remasking_strategy='low_confidence_dynamic',
#         confidence_threshold=0.85,
#         eb_threshold=None,
#         stopping_criteria_idx=None,
#         causal_prefill=False
#     ):
#     """
#     Batched block diffusion generation with Dream-style logit shift.

#     Processes multiple prompts in parallel for better GPU utilization.
#     Uses the same KV cache mechanism as the single-sample version:
#       - store_kv=False during denoising (read-only cache)
#       - store_kv=True after block is finalized (cache grows by block_length)

#     Prompts are padded to a common length. Blocks in the padding region
#     between shorter prompts and the max prompt length are handled naturally:
#     real prompt tokens have mask_index=False so they pass through denoising
#     untouched, and their KV is stored when the block finalizes.

#     Args:
#         prompts: list of dicts, each with 'input_ids' tensor of shape (1, seq_len)
#         causal_prefill: If True, use regular causal attention for the prompt
#             prefill (matching training where clean prefix uses causal attention).
#             If False, use block-causal attention for prefill (legacy behavior).
#         Other args same as block_diffusion_generate_with_shift.

#     Returns:
#         x: tensor of shape (batch, total_length) with generated tokens
#         prompt_lengths: list of original prompt lengths per sample
#     """

#     model.eval()
#     batch_size = len(prompts)
#     device = model.device

#     # Collect prompt lengths
#     prompt_lengths = [p['input_ids'].shape[1] for p in prompts]
#     max_prompt_length = max(prompt_lengths)

#     # Compute total sequence length (padded to multiple of block_length)
#     num_blocks = (max_prompt_length + gen_length +
#                   block_length - 1) // block_length
#     total_length = num_blocks * block_length

#     # Build batched input: (batch, total_length), filled with mask_id
#     x = torch.full((batch_size, total_length), mask_id,
#                    dtype=torch.long, device=device)
#     for i, p in enumerate(prompts):
#         x[i, :prompt_lengths[i]] = p['input_ids'][0]

#     # Attention mask: block-causal, shared across batch (broadcasts over dim 0)
#     block_mask = torch.tril(torch.ones(
#         num_blocks, num_blocks, device=device))
#     block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
#                                                .repeat_interleave(block_length, dim=1).unsqueeze(0)
#     position_ids = torch.arange(total_length, device=device).unsqueeze(0)

#     # Original: prefill only blocks where ALL samples have real tokens
#     min_prompt_length = min(prompt_lengths)
#     prefill_blocks = min_prompt_length // block_length
#     prefill_length = prefill_blocks * block_length

#     # New: prefill up to max prompt length so all real prompt tokens get
#     # correct causal attention (matching training). Short prompts have
#     # mask tokens in the padding region — harmless in the KV cache.
#     # prefill_blocks = max_prompt_length // block_length
#     # prefill_length = prefill_blocks * block_length

#     past_key_values = DynamicCache()
#     prev_block_last_logits = None

#     if prefill_length > 0:
#         cur_x = x[:, :prefill_length]
#         if causal_prefill:
#             # Regular causal attention for prompt (matches training where
#             # clean prefix uses standard causal, not block-causal)
#             cur_attn_mask = torch.tril(torch.ones(
#                 prefill_length, prefill_length, device=device)).unsqueeze(0)
#         else:
#             cur_attn_mask = block_diffusion_attention_mask[:,
#                                                            :prefill_length, :prefill_length]
#         cur_position_ids = position_ids[:, :prefill_length]
#         prefill_output = model(cur_x,
#               attention_mask=cur_attn_mask,
#               position_ids=cur_position_ids,
#               past_key_values=past_key_values,
#               use_cache=True,
#               store_kv=True)
#         prev_block_last_logits = prefill_output.logits[:, -1:, :]  # (batch, 1, vocab)

#     num_transfer_tokens = get_num_transfer_tokens(
#         block_length, denoising_steps)

#     # Per-sample stopping
#     finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

#     # Decode stage
#     for num_block in range(prefill_blocks, num_blocks):
#         if finished.all():
#             break

#         block_start = num_block * block_length
#         block_end = (num_block + 1) * block_length
#         cur_x = x[:, block_start:block_end].clone()
#         cur_attn_mask = block_diffusion_attention_mask[
#             :, block_start:block_end, :block_end
#         ]
#         cur_position_ids = position_ids[:, block_start:block_end]

#         # For finished samples, replace masks with a non-mask token
#         # so denoising is a no-op for them
#         if finished.any():
#             fill_token = stopping_criteria_idx[0] if stopping_criteria_idx else 0
#             for i in range(batch_size):
#                 if finished[i]:
#                     cur_x[i] = fill_token

#         # Denoising steps
#         for step in range(denoising_steps):
#             mask_index = (cur_x == mask_id)
#             if not mask_index.any():
#                 break  # Entire batch has no masks in this block

#             logits = model(cur_x,
#                            attention_mask=cur_attn_mask,
#                            position_ids=cur_position_ids,
#                            past_key_values=past_key_values,
#                            use_cache=True,
#                            store_kv=False).logits

#             # Dream-style logit shift
#             if prev_block_last_logits is not None:
#                 logits = torch.cat([prev_block_last_logits, logits[:, :-1]], dim=1)
#             else:
#                 logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

#             # Sampling
#             x0, x0_p = sample_with_temperature_topk_topp(
#                 logits,
#                 temperature=temperature,
#                 top_k=top_k,
#                 top_p=top_p
#             )

#             # Remasking strategy
#             if remasking_strategy == 'sequential':
#                 transfer_index = torch.zeros_like(x0, dtype=torch.bool)
#                 for j in range(batch_size):
#                     if not mask_index[j].any():
#                         continue
#                     first_mask = mask_index[j].nonzero(as_tuple=True)[0].min().item()
#                     num_masked = mask_index[j].sum().item()
#                     k = min(num_transfer_tokens[step].item(), num_masked)
#                     transfer_index[j, first_mask:first_mask + k] = True

#             elif remasking_strategy == 'low_confidence_static':
#                 confidence = torch.where(mask_index, x0_p, -torch.inf)
#                 transfer_index = torch.zeros_like(x0, dtype=torch.bool)
#                 for j in range(batch_size):
#                     if not mask_index[j].any():
#                         continue
#                     num_masked = mask_index[j].sum().item()
#                     k = min(num_transfer_tokens[step].item(), num_masked)
#                     _, idx = torch.topk(confidence[j], k)
#                     transfer_index[j, idx] = True

#             elif remasking_strategy == 'low_confidence_dynamic':
#                 confidence = torch.where(mask_index, x0_p, -torch.inf)
#                 transfer_index = torch.zeros_like(x0, dtype=torch.bool)
#                 for j in range(batch_size):
#                     if not mask_index[j].any():
#                         continue
#                     high_conf_mask = confidence[j] > confidence_threshold
#                     num_high_confidence = high_conf_mask.sum()
#                     if num_high_confidence >= num_transfer_tokens[step]:
#                         transfer_index[j] = high_conf_mask
#                     else:
#                         num_masked = mask_index[j].sum().item()
#                         k = min(num_transfer_tokens[step].item(), num_masked)
#                         _, idx = torch.topk(confidence[j], k)
#                         transfer_index[j, idx] = True

#             else:
#                 raise ValueError(
#                     f"Unknown remasking strategy: {remasking_strategy}")

#             # Only transfer at masked positions (safety guard for batching)
#             transfer_index = transfer_index & mask_index
#             cur_x[transfer_index] = x0[transfer_index]

#         # Always store KV after denoising (cache grows by block_length)
#         store_output = model(cur_x,
#               attention_mask=cur_attn_mask,
#               position_ids=cur_position_ids,
#               past_key_values=past_key_values,
#               use_cache=True,
#               store_kv=True)
#         prev_block_last_logits = store_output.logits[:, -1:, :]

#         # Write back denoised block
#         x[:, block_start:block_end] = cur_x

#         # Check EOS per sample
#         if stopping_criteria_idx is not None:
#             for i in range(batch_size):
#                 if not finished[i]:
#                     gen_region = x[i, prompt_lengths[i]:]
#                     if any(stop_idx in gen_region for stop_idx in stopping_criteria_idx):
#                         finished[i] = True

#     return x, prompt_lengths
# End revision by Yifan Yu
# ============================================================================


# ============================================================================
# Speculative block-2 generation with Dream-style logit shift
# Modified by Yifan Yu
#
# AR generation that speculatively predicts 2 tokens per forward pass.
# Each forward: input [t_prev, MASK] with causal attention where t_prev
# does NOT attend to MASK (so t_prev's KV is always correct).
#
# Dream shift gives us:
#   shifted_logits[1] = logits[0] -> predicts MASK position = t_diff
#   logits[1] -> predicts next position (carry) = t_carry
#
# Accept (t_carry conf >= threshold): advance 2, defer t_diff commit.
# Reject: advance 1, no pending.
# MASK KV always trimmed. KV cache always clean.
# True TPF = 1 + accept_rate.
# ============================================================================
@torch.no_grad()
def causal_block2_generate_with_shift(
        model,
        prompt,
        mask_id,
        gen_length=128,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        confidence_threshold=0.85,
        stopping_criteria_idx=None,
    ):
    """
    Block-2 generation with deferred clean KV commit.

    Each forward: [pending..., t_prev, MASK] (2 or 3 tokens).
    MASK KV always trimmed. On accept, t_diff's commit is deferred to next forward.
    KV cache is always clean.

    Returns:
        x: tensor of shape (1, prompt_length + num_generated) with output tokens.
    """
    model.eval()
    device = model.device
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]

    def _trim_kv(past_kv, n=1):
        for layer_idx in range(len(past_kv)):
            if hasattr(past_kv, 'key_cache'):
                past_kv.key_cache[layer_idx] = past_kv.key_cache[layer_idx][:, :, :-n, :]
                past_kv.value_cache[layer_idx] = past_kv.value_cache[layer_idx][:, :, :-n, :]
            elif hasattr(past_kv, 'layers'):
                layer = past_kv.layers[layer_idx]
                layer.keys = layer.keys[:, :, :-n, :]
                layer.values = layer.values[:, :, :-n, :]

    # --- Causal prefill ---
    past_key_values = DynamicCache()
    pos_ids = torch.arange(prompt_length, device=device).unsqueeze(0)
    causal_mask = torch.tril(torch.ones(
        prompt_length, prompt_length, device=device)).unsqueeze(0)

    prefill_output = model(
        input_ids,
        attention_mask=causal_mask,
        position_ids=pos_ids,
        past_key_values=past_key_values,
        use_cache=True,
        store_kv=True)

    prev_last_logits = prefill_output.logits[:, -1:, :]
    first_token, _ = sample_with_temperature_topk_topp(
        prev_last_logits, temperature, top_k, top_p)

    # --- Decode loop with deferred commit ---
    generated_tokens = [first_token.squeeze(0)]
    current_pos = prompt_length
    t_prev = first_token.squeeze().item()
    pending_token = None
    pending_pos = None

    while current_pos + 1 < prompt_length + gen_length:
        # Build input: [pending, t_prev, MASK] or [t_prev, MASK]
        if pending_token is not None:
            cur_tokens = [pending_token, t_prev, mask_id]
            cur_positions = [pending_pos, current_pos, current_pos + 1]
            n_input = 3
        else:
            cur_tokens = [t_prev, mask_id]
            cur_positions = [current_pos, current_pos + 1]
            n_input = 2

        cur_x = torch.tensor([cur_tokens], dtype=torch.long, device=device)
        cur_pos_ids = torch.tensor([cur_positions], dtype=torch.long, device=device)
        kv_len = cur_positions[0]

        # Causal attention within new tokens
        cur_attn_mask = torch.ones(1, n_input, kv_len + n_input, device=device)
        for i in range(n_input):
            for j in range(i + 1, n_input):
                cur_attn_mask[:, i, kv_len + j] = 0

        output = model(
            cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_pos_ids,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True)
        logits = output.logits

        # Always trim MASK KV
        _trim_kv(past_key_values, n=1)

        # Dream shift
        shifted_logits = torch.cat([prev_last_logits, logits[:, :-1]], dim=1)

        # t_diff: from clean hidden(t_prev), always committed
        t_diff, t_diff_prob = sample_with_temperature_topk_topp(
            shifted_logits[:, -1:, :], temperature, top_k, top_p)
        t_diff_val = t_diff.squeeze().item()

        # t_carry: from MASK hidden, speculative
        t_carry, t_carry_prob = sample_with_temperature_topk_topp(
            logits[:, -1:, :], temperature, top_k, top_p)
        t_carry_val = t_carry.squeeze().item()
        t_carry_conf = t_carry_prob.squeeze().item()

        if t_carry_conf >= confidence_threshold:
            # ACCEPT: defer t_diff commit, advance 2
            prev_last_logits = logits[:, -1:, :]
            generated_tokens.append(t_diff.squeeze(0))
            generated_tokens.append(t_carry.squeeze(0))
            pending_token = t_diff_val
            pending_pos = current_pos + 1
            current_pos += 2
            t_prev = t_carry_val

            if stopping_criteria_idx is not None:
                if t_diff_val in stopping_criteria_idx or t_carry_val in stopping_criteria_idx:
                    break
        else:
            # REJECT: no pending, advance 1
            prev_last_logits = logits[:, -2:-1, :]
            generated_tokens.append(t_diff.squeeze(0))
            pending_token = None
            pending_pos = None
            current_pos += 1
            t_prev = t_diff_val

            if stopping_criteria_idx is not None:
                if t_diff_val in stopping_criteria_idx:
                    break

    all_generated = torch.cat(generated_tokens, dim=0)
    x = torch.cat([input_ids.squeeze(0), all_generated], dim=0).unsqueeze(0)
    return x
# End revision by Yifan Yu
# ============================================================================


# ============================================================================
# Block-2 generation with deferred commit + verification
# Modified by Yifan Yu
#
# Same as causal_block2_generate_with_shift but verifies t_carry when pending
# exists. The 3-token forward [t_diff_pending, t_carry, MASK] computes
# hidden(t_diff_pending) which gives the clean prediction for t_carry's position.
# If t_carry has low probability under this clean distribution, replace it.
# ============================================================================
@torch.no_grad()
def causal_block2_verified_generate_with_shift(
        model,
        prompt,
        mask_id,
        gen_length=128,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        confidence_threshold=0.85,
        verify_threshold=0.3,
        stopping_criteria_idx=None,
    ):
    """
    Block-2 generation with deferred clean KV commit + verification.

    Same as causal_block2 but when pending exists (3-token forward), verifies
    t_carry against clean prediction from hidden(t_diff_pending). If t_carry
    has low clean probability, resample from clean distribution and retry.

    Returns:
        x: tensor of shape (1, prompt_length + num_generated) with output tokens.
    """
    model.eval()
    device = model.device
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]

    def _trim_kv(past_kv, n=1):
        for layer_idx in range(len(past_kv)):
            if hasattr(past_kv, 'key_cache'):
                past_kv.key_cache[layer_idx] = past_kv.key_cache[layer_idx][:, :, :-n, :]
                past_kv.value_cache[layer_idx] = past_kv.value_cache[layer_idx][:, :, :-n, :]
            elif hasattr(past_kv, 'layers'):
                layer = past_kv.layers[layer_idx]
                layer.keys = layer.keys[:, :, :-n, :]
                layer.values = layer.values[:, :, :-n, :]

    # --- Causal prefill ---
    past_key_values = DynamicCache()
    pos_ids = torch.arange(prompt_length, device=device).unsqueeze(0)
    causal_mask = torch.tril(torch.ones(
        prompt_length, prompt_length, device=device)).unsqueeze(0)

    prefill_output = model(
        input_ids,
        attention_mask=causal_mask,
        position_ids=pos_ids,
        past_key_values=past_key_values,
        use_cache=True,
        store_kv=True)

    prev_last_logits = prefill_output.logits[:, -1:, :]
    first_token, _ = sample_with_temperature_topk_topp(
        prev_last_logits, temperature, top_k, top_p)

    # --- Decode loop with deferred commit + verification ---
    generated_tokens = [first_token.squeeze(0)]
    current_pos = prompt_length
    t_prev = first_token.squeeze().item()
    pending_token = None
    pending_pos = None

    while current_pos + 1 < prompt_length + gen_length:
        # Build input
        if pending_token is not None:
            cur_tokens = [pending_token, t_prev, mask_id]
            cur_positions = [pending_pos, current_pos, current_pos + 1]
            n_input = 3
        else:
            cur_tokens = [t_prev, mask_id]
            cur_positions = [current_pos, current_pos + 1]
            n_input = 2

        cur_x = torch.tensor([cur_tokens], dtype=torch.long, device=device)
        cur_pos_ids = torch.tensor([cur_positions], dtype=torch.long, device=device)
        kv_len = cur_positions[0]

        cur_attn_mask = torch.ones(1, n_input, kv_len + n_input, device=device)
        for i in range(n_input):
            for j in range(i + 1, n_input):
                cur_attn_mask[:, i, kv_len + j] = 0

        output = model(
            cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_pos_ids,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True)
        logits = output.logits

        # --- Verification (only when pending exists) ---
        if pending_token is not None:
            # logits[0] = hidden(t_diff_pending) → clean prediction for t_carry's position
            clean_probs = torch.softmax(logits[:, 0, :], dim=-1)
            t_carry_clean_prob = clean_probs[0, t_prev].item()

            if t_carry_clean_prob < verify_threshold:
                # VERIFY FAIL: trim t_carry + MASK KV, resample, retry
                _trim_kv(past_key_values, n=2)
                new_t_carry, _ = sample_with_temperature_topk_topp(
                    logits[:, 0:1, :], temperature, top_k, top_p)
                new_t_carry_val = new_t_carry.squeeze().item()
                generated_tokens[-1] = new_t_carry.squeeze(0)
                prev_last_logits = logits[:, 0:1, :]
                t_prev = new_t_carry_val
                pending_token = None
                pending_pos = None
                continue

        # Always trim MASK KV
        _trim_kv(past_key_values, n=1)

        # Dream shift
        shifted_logits = torch.cat([prev_last_logits, logits[:, :-1]], dim=1)

        t_diff, t_diff_prob = sample_with_temperature_topk_topp(
            shifted_logits[:, -1:, :], temperature, top_k, top_p)
        t_diff_val = t_diff.squeeze().item()

        t_carry, t_carry_prob = sample_with_temperature_topk_topp(
            logits[:, -1:, :], temperature, top_k, top_p)
        t_carry_val = t_carry.squeeze().item()
        t_carry_conf = t_carry_prob.squeeze().item()

        if t_carry_conf >= confidence_threshold:
            # ACCEPT
            prev_last_logits = logits[:, -1:, :]
            generated_tokens.append(t_diff.squeeze(0))
            generated_tokens.append(t_carry.squeeze(0))
            pending_token = t_diff_val
            pending_pos = current_pos + 1
            current_pos += 2
            t_prev = t_carry_val

            if stopping_criteria_idx is not None:
                if t_diff_val in stopping_criteria_idx or t_carry_val in stopping_criteria_idx:
                    break
        else:
            # REJECT
            prev_last_logits = logits[:, -2:-1, :]
            generated_tokens.append(t_diff.squeeze(0))
            pending_token = None
            pending_pos = None
            current_pos += 1
            t_prev = t_diff_val

            if stopping_criteria_idx is not None:
                if t_diff_val in stopping_criteria_idx:
                    break

    all_generated = torch.cat(generated_tokens, dim=0)
    x = torch.cat([input_ids.squeeze(0), all_generated], dim=0).unsqueeze(0)
    return x
# End revision by Yifan Yu
# ============================================================================


# ============================================================================
# Block-2 generation with deferred commit + speculative decoding verification
# Modified by Yifan Yu
#
# Instead of a hard verify_threshold, uses the speculative decoding acceptance
# criterion: accept t_carry with probability min(1, p(x)/q(x)) where
#   q(x) = draft probability (from MASK hidden state)
#   p(x) = clean probability (from t_diff_pending hidden state)
# On rejection, resample from corrected distribution max(0, p - q).
# This guarantees the output follows the clean (target) distribution.
# ============================================================================
@torch.no_grad()
def causal_block2_spec_verified_generate_with_shift(
        model,
        prompt,
        mask_id,
        gen_length=128,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        confidence_threshold=0.85,
        stopping_criteria_idx=None,
    ):
    """
    Block-2 generation with deferred clean KV commit + speculative decoding
    verification.

    Same as causal_block2 but when pending exists (3-token forward), verifies
    t_carry using the speculative decoding acceptance criterion:
        r = p(x) / q(x)
        if r >= 1: always accept
        if r <  1: accept with probability r, else resample from max(0, p-q)

    The draft prob q(x) is stored from the previous step's MASK logits.
    The clean prob p(x) comes from hidden(t_diff_pending) in the current step.

    Returns:
        x: tensor of shape (1, prompt_length + num_generated) with output tokens.
    """
    model.eval()
    device = model.device
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]

    def _trim_kv(past_kv, n=1):
        for layer_idx in range(len(past_kv)):
            if hasattr(past_kv, 'key_cache'):
                past_kv.key_cache[layer_idx] = past_kv.key_cache[layer_idx][:, :, :-n, :]
                past_kv.value_cache[layer_idx] = past_kv.value_cache[layer_idx][:, :, :-n, :]
            elif hasattr(past_kv, 'layers'):
                layer = past_kv.layers[layer_idx]
                layer.keys = layer.keys[:, :, :-n, :]
                layer.values = layer.values[:, :, :-n, :]

    # --- Causal prefill ---
    past_key_values = DynamicCache()
    pos_ids = torch.arange(prompt_length, device=device).unsqueeze(0)
    causal_mask = torch.tril(torch.ones(
        prompt_length, prompt_length, device=device)).unsqueeze(0)

    prefill_output = model(
        input_ids,
        attention_mask=causal_mask,
        position_ids=pos_ids,
        past_key_values=past_key_values,
        use_cache=True,
        store_kv=True)

    prev_last_logits = prefill_output.logits[:, -1:, :]
    first_token, _ = sample_with_temperature_topk_topp(
        prev_last_logits, temperature, top_k, top_p)

    # --- Decode loop with deferred commit + spec verification ---
    generated_tokens = [first_token.squeeze(0)]
    current_pos = prompt_length
    t_prev = first_token.squeeze().item()
    pending_token = None
    pending_pos = None
    # Store draft distribution for spec decoding verification
    pending_draft_probs = None  # full draft distribution q(x) for t_carry

    while current_pos + 1 < prompt_length + gen_length:
        # Build input
        if pending_token is not None:
            cur_tokens = [pending_token, t_prev, mask_id]
            cur_positions = [pending_pos, current_pos, current_pos + 1]
            n_input = 3
        else:
            cur_tokens = [t_prev, mask_id]
            cur_positions = [current_pos, current_pos + 1]
            n_input = 2

        cur_x = torch.tensor([cur_tokens], dtype=torch.long, device=device)
        cur_pos_ids = torch.tensor([cur_positions], dtype=torch.long, device=device)
        kv_len = cur_positions[0]

        cur_attn_mask = torch.ones(1, n_input, kv_len + n_input, device=device)
        for i in range(n_input):
            for j in range(i + 1, n_input):
                cur_attn_mask[:, i, kv_len + j] = 0

        output = model(
            cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_pos_ids,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True)
        logits = output.logits

        # --- Speculative decoding verification (only when pending exists) ---
        if pending_token is not None:
            # clean distribution p(x) from hidden(t_diff_pending)
            clean_logits = logits[:, 0, :]
            if temperature is not None and temperature > 0 and temperature != 1.0:
                clean_logits = clean_logits / temperature
            clean_probs = torch.softmax(clean_logits, dim=-1)  # p(x)

            # draft distribution q(x) was stored from previous step
            draft_probs = pending_draft_probs  # q(x)

            # p(x) and q(x) for the accepted t_carry token
            p_x = clean_probs[0, t_prev].item()
            q_x = draft_probs[0, t_prev].item()

            # Speculative decoding acceptance: accept with prob min(1, p/q)
            r = p_x / q_x if q_x > 0 else 0.0
            accept = r >= 1.0 or torch.rand(1, device=device).item() < r

            if not accept:
                # REJECT: trim t_carry + MASK KV, resample from corrected dist
                _trim_kv(past_key_values, n=2)

                # Corrected distribution: norm(max(0, p - q))
                corrected = torch.clamp(clean_probs - draft_probs, min=0)
                corrected_sum = corrected.sum()
                if corrected_sum > 0:
                    corrected = corrected / corrected_sum
                    new_t_carry = torch.multinomial(corrected, num_samples=1)
                else:
                    # Fallback: sample from clean distribution
                    new_t_carry = torch.multinomial(clean_probs, num_samples=1)

                new_t_carry_val = new_t_carry.squeeze().item()
                generated_tokens[-1] = new_t_carry.squeeze(0)
                prev_last_logits = logits[:, 0:1, :]
                t_prev = new_t_carry_val
                pending_token = None
                pending_pos = None
                pending_draft_probs = None

                if stopping_criteria_idx is not None:
                    if new_t_carry_val in stopping_criteria_idx:
                        break
                continue

        # Always trim MASK KV
        _trim_kv(past_key_values, n=1)

        # Dream shift
        shifted_logits = torch.cat([prev_last_logits, logits[:, :-1]], dim=1)

        t_diff, t_diff_prob = sample_with_temperature_topk_topp(
            shifted_logits[:, -1:, :], temperature, top_k, top_p)
        t_diff_val = t_diff.squeeze().item()

        t_carry, t_carry_prob = sample_with_temperature_topk_topp(
            logits[:, -1:, :], temperature, top_k, top_p)
        t_carry_val = t_carry.squeeze().item()
        t_carry_conf = t_carry_prob.squeeze().item()

        if t_carry_conf >= confidence_threshold:
            # ACCEPT: defer t_diff commit, advance 2
            # Store draft distribution for verification in next step
            carry_logits = logits[:, -1, :]
            if temperature is not None and temperature > 0 and temperature != 1.0:
                carry_logits = carry_logits / temperature
            pending_draft_probs = torch.softmax(carry_logits, dim=-1)

            prev_last_logits = logits[:, -1:, :]
            generated_tokens.append(t_diff.squeeze(0))
            generated_tokens.append(t_carry.squeeze(0))
            pending_token = t_diff_val
            pending_pos = current_pos + 1
            current_pos += 2
            t_prev = t_carry_val

            if stopping_criteria_idx is not None:
                if t_diff_val in stopping_criteria_idx or t_carry_val in stopping_criteria_idx:
                    break
        else:
            # REJECT: no pending, advance 1
            prev_last_logits = logits[:, -2:-1, :]
            generated_tokens.append(t_diff.squeeze(0))
            pending_token = None
            pending_pos = None
            pending_draft_probs = None
            current_pos += 1
            t_prev = t_diff_val

            if stopping_criteria_idx is not None:
                if t_diff_val in stopping_criteria_idx:
                    break

    all_generated = torch.cat(generated_tokens, dim=0)
    x = torch.cat([input_ids.squeeze(0), all_generated], dim=0).unsqueeze(0)
    return x
# End spec verified revision by Yifan Yu
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the pretrained model directory")
    parser.add_argument("--trust_remote_code", action='store_true')
    parser.add_argument("--mask_id", type=int, default=None,
                        help="Mask token id for Diffusion")
    parser.add_argument("--prompt_length", type=int, default=4096,
                        help="Maximum prompt length in tokens")
    parser.add_argument("--gen_length", type=int, default=20480,
                        help="Maximum generation length in tokens")
    parser.add_argument("--block_length", type=int, default=4,
                        help="Length of token block to replace each denoising step")
    parser.add_argument("--denoising_steps", type=int, default=4,
                        help="Number of denoising steps (iterations)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-K sampling (0 to disable)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-P sampling probability threshold")
    parser.add_argument("--remasking_strategy", type=str, default="low_confidence_dynamic",
                        choices=["low_confidence_dynamic",
                                 "low_confidence_static",
                                 "sequential",
                                 "entropy_bounded"],
                        help="Strategy for remasking tokens")
    parser.add_argument("--confidence_threshold", type=float, default=0.85,
                        help="Confidence threshold for low-confidence remasking")
    parser.add_argument("--eb_threshold", type=float, default=0.35,
                        help="entropy threshold for entropy bounded sampling")
    parser.add_argument("--stopping_criteria_idx", type=int, nargs="+", default=None,
                        help="List of token IDs that stop generation (e.g. eos_token_id)")
    
    # Sliding window arguments
    parser.add_argument("--sliding_window", action='store_true',
                        help="Enable sliding window block diffusion (Ideas 2+3)")
    parser.add_argument("--enable_revision", action='store_true', default=True,
                        help="Enable token revision in sliding window mode (Idea 3)")
    parser.add_argument("--no_revision", action='store_true',
                        help="Disable token revision (only Idea 2, no Idea 3)")

    parser.add_argument("--device", type=str, default="cuda",)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"],)
    
    args = parser.parse_args()
    
    # Handle revision flag
    if args.no_revision:
        args.enable_revision = False
    
    if args.remasking_strategy == "low_confidence_dynamic" and args.confidence_threshold is None:
        parser.error(
            "--confidence_threshold is required when --remasking_strategy=low_confidence_dynamic"
        )
    if args.remasking_strategy == "entropy_bounded" and args.eb_threshold is None:
        parser.error(
            "--eb_threshold is required when --remasking_strategy=entropy_bounded"
        )
    if args.sliding_window and args.block_length % 2 != 0:
        parser.error(
            "--block_length must be even when using --sliding_window"
        )
    return args


if __name__ == "__main__":
    args = parse_args()

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.dtype,
        device_map=args.device
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
    )

    if args.mask_id is None:
        args.mask_id = tokenizer(tokenizer.mask_token)['input_ids'][0]
    if args.stopping_criteria_idx is None:
        gen_cfg = GenerationConfig.from_pretrained(args.model_dir,)
        args.stopping_criteria_idx = gen_cfg.eos_token_id
    if isinstance(args.stopping_criteria_idx, int):
        args.stopping_criteria_idx = [args.stopping_criteria_idx,]
    args.stop_words = tokenizer.convert_ids_to_tokens(
        args.stopping_criteria_idx)
    print(f"Your Arguments: {args}")

    origin_prompt = [
        # dict(role="user", content="Given the function $f(x) = \\frac{4x^2 - 4x + 4}{x^2 + 2x + 4}$, where $x \\in \\mathbb{R}$, determine its minimum value.\nPlease reason step by step, and put your final answer within \\boxed{}.\n"),
        dict(role="user", content="If the domain of the function $\\log x^2$ is $x < a$ or $x > b$, for some $a$ and $b$, find $a + b$.\nPlease reason step by step, and put your final answer within \\boxed{}.\n")
    ]

    messages = tokenizer.apply_chat_template(
        origin_prompt, add_generation_prompt=True, tokenize=False)
    tokenize_kwargs = dict(
        return_tensors='pt',
        padding=True,
        truncation=True,
        add_special_tokens=False,
        max_length=args.prompt_length
    )

    tokens = tokenizer.batch_encode_plus([messages], **tokenize_kwargs)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    if args.sliding_window:
        print(f"Using sliding window block diffusion (revision={'enabled' if args.enable_revision else 'disabled'})")
        output_ids = sliding_window_block_diffusion_generate(
            model,
            prompt=tokens,
            mask_id=args.mask_id,
            gen_length=args.gen_length,
            block_length=args.block_length,
            denoising_steps=args.denoising_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            remasking_strategy=args.remasking_strategy,
            confidence_threshold=args.confidence_threshold,
            eb_threshold=args.eb_threshold,
            enable_revision=args.enable_revision,
            stopping_criteria_idx=args.stopping_criteria_idx
        )
    else:
        output_ids = block_diffusion_generate(
            model,
            prompt=tokens,
            mask_id=args.mask_id,
            gen_length=args.gen_length,
            block_length=args.block_length,
            denoising_steps=args.denoising_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            remasking_strategy=args.remasking_strategy,
            confidence_threshold=args.confidence_threshold,
            eb_threshold=args.eb_threshold,
            stopping_criteria_idx=args.stopping_criteria_idx
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    cleaned_text = output_text.replace('<|MASK|>', '')
    print(cleaned_text)
