import gc
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sentence_transformers.util import (semantic_search,
                                        dot_score,
                                        normalize_embeddings)


def map_tokens_between_models(tokens, tokenizer_1, tokenizer_2):
    """
    Map tokens from model_1's vocabulary to the closest tokens in model_2's vocabulary.
    This is necessary when models have different vocabularies.
    """
    # Convert token IDs to strings using model_1's tokenizer
    if hasattr(tokenizer_1, "convert_ids_to_tokens"):
        # For transformers tokenizers
        token_strings = tokenizer_1.convert_ids_to_tokens(tokens.tolist())
    else:
        # Fallback for other tokenizers
        token_strings = [str(t) for t in tokens.tolist()]

    # Convert strings to token IDs using model_2's tokenizer
    mapped_tokens = []
    for token_str in token_strings:
        if hasattr(tokenizer_2, "convert_tokens_to_ids"):
            # For transformers tokenizers
            mapped_id = tokenizer_2.convert_tokens_to_ids(token_str)
            # Use UNK token if not found
            if mapped_id == tokenizer_2.unk_token_id:
                mapped_id = tokenizer_2.unk_token_id if hasattr(tokenizer_2, "unk_token_id") else 0
        else:
            # Fallback for other tokenizers - map to token 0 (usually padding or unknown)
            mapped_id = 0

        mapped_tokens.append(mapped_id)

    return torch.tensor(mapped_tokens, device=tokens.device)


def token_gradients_agg(model_1, model_2, tokenizer_1, tokenizer_2, init):
    # Make sure init is on the correct device
    if not init.is_cuda:
        init = init.to(model_1.device)

    # Get embedding weights
    embed_weights_1 = model_1.get_input_embeddings().weight
    embed_weights_2 = model_2.get_input_embeddings().weight

    # Get vocab sizes
    vocab_size_1 = embed_weights_1.shape[0]
    vocab_size_2 = embed_weights_2.shape[0]

    print(f"Model 1 vocab size: {vocab_size_1}")
    print(f"Model 2 vocab size: {vocab_size_2}")
    print(f"Max token ID in init: {init.max().item()}")

    # Create valid tokens for each model
    valid_tokens_1 = torch.clamp(init, 0, vocab_size_1 - 1)

    # Handle vocabulary mapping for model_2
    # Option 1: Try to map tokens between vocabularies
    try:
        valid_tokens_2 = map_tokens_between_models(init, tokenizer_1, tokenizer_2)
        print(f"Mapped tokens for model_2: {valid_tokens_2.tolist()}")
    except Exception as e:
        print(f"Error mapping tokens: {e}")
        # Option 2: Fallback to simple clamping
        valid_tokens_2 = torch.clamp(init, 0, vocab_size_2 - 1)
        print(f"Clamped tokens for model_2: {valid_tokens_2.tolist()}")

    # Create one-hot encodings
    one_hot_1 = torch.zeros(
        init.shape[0],
        vocab_size_1,
        device=model_1.device,
        dtype=embed_weights_1.dtype
    )

    one_hot_1.scatter_(
        1,
        valid_tokens_1.unsqueeze(1),
        torch.ones(one_hot_1.shape[0], 1, device=model_1.device, dtype=embed_weights_1.dtype)
    )
    one_hot_1.requires_grad_()

    one_hot_2 = torch.zeros(
        init.shape[0],
        vocab_size_2,
        device=model_2.device,
        dtype=embed_weights_2.dtype
    )

    one_hot_2.scatter_(
        1,
        valid_tokens_2.unsqueeze(1),
        torch.ones(one_hot_2.shape[0], 1, device=model_2.device, dtype=embed_weights_2.dtype)
    )
    one_hot_2.requires_grad_()

    # Forward pass for model 1
    try:
        input_embeds_1 = (one_hot_1 @ embed_weights_1).unsqueeze(0)
        logits_1 = model_1(inputs_embeds=input_embeds_1).logits
        shift_logits_1 = logits_1[..., -1, :].contiguous()
        loss_1 = nn.CrossEntropyLoss()(
            shift_logits_1.view(-1, shift_logits_1.size(-1)),
            torch.tensor([tokenizer_1.eos_token_id], device=model_1.device)
        )
    except RuntimeError as e:
        print(f"Error in model_1 forward pass: {e}")
        print(f"one_hot_1 shape: {one_hot_1.shape}, embed_weights_1 shape: {embed_weights_1.shape}")
        print(f"input_embeds_1 shape: {input_embeds_1.shape if 'input_embeds_1' in locals() else 'not created'}")
        raise

    # Forward pass for model 2
    try:
        input_embeds_2 = (one_hot_2 @ embed_weights_2).unsqueeze(0)
        logits_2 = model_2(inputs_embeds=input_embeds_2).logits
        shift_logits_2 = logits_2[..., -1, :].contiguous()
        loss_2 = nn.CrossEntropyLoss()(
            shift_logits_2.view(-1, shift_logits_2.size(-1)),
            torch.tensor([tokenizer_2.eos_token_id], device=model_2.device)
        )
    except RuntimeError as e:
        print(f"Error in model_2 forward pass: {e}")
        print(f"one_hot_2 shape: {one_hot_2.shape}, embed_weights_2 shape: {embed_weights_2.shape}")
        print(f"input_embeds_2 shape: {input_embeds_2.shape if 'input_embeds_2' in locals() else 'not created'}")
        raise

    # Combined loss
    loss = (loss_1 + loss_2) / 2
    loss.backward()

    return one_hot_1.grad.clone()


def sample_control_agg(model_1, model_2, tokenizer_1, tokenizer_2, init, grad, batch_size, topk=5, topk_semanteme=10):
    # Make sure init is on the right device
    if not init.is_cuda:
        init = init.to(model_1.device)

    # Get vocabulary sizes
    vocab_size_1 = model_1.get_input_embeddings().weight.shape[0]
    vocab_size_2 = model_2.get_input_embeddings().weight.shape[0]

    # Get normalized embeddings
    curr_embeds_1 = model_1.get_input_embeddings()(torch.clamp(init, 0, vocab_size_1 - 1).unsqueeze(0))
    curr_embeds_1 = torch.nn.functional.normalize(curr_embeds_1, dim=2)

    embedding_matrix_1 = model_1.get_input_embeddings().weight
    embedding_matrix_1 = normalize_embeddings(embedding_matrix_1)

    token_length = init.shape[0]

    top_indices = torch.zeros(token_length, topk, dtype=torch.long, device=init.device)

    for i in range(token_length):
        similar = []
        temp = []
        query = curr_embeds_1[0][i]
        corpus = embedding_matrix_1

        hits = semantic_search(query,
                               corpus,
                               top_k=min(topk_semanteme + 1, vocab_size_1),
                               score_function=dot_score)

        for hit in hits:
            for dic in hit[1:]:  # Skip the first hit (same token)
                corpus_id = dic["corpus_id"]
                if 0 <= corpus_id < vocab_size_1:  # Ensure index is in bounds
                    temp.append((grad[i][corpus_id].item(), corpus_id))

        # Sort by gradient value (ascending)
        temp.sort()

        # Add safety check
        if len(temp) < topk:
            # If we don't have enough valid tokens, repeat the last one
            while len(temp) < topk:
                if len(temp) > 0:
                    temp.append(temp[-1])
                else:
                    # If we have no valid tokens, use token 0 as fallback
                    temp.append((0.0, 0))

        # Take top-k tokens
        similar = [temp[j][1] for j in range(min(topk, len(temp)))]

        # Convert to tensor and assign
        top_indices[i] = torch.tensor(similar, dtype=torch.long, device=init.device)

    # Create batch of candidates with one token changed per candidate
    original_control_toks = init.repeat(batch_size, 1)

    # Specify which position to modify for each candidate
    # Ensure we don't go out of bounds with new_token_pos
    new_token_pos = torch.arange(
        1,  # Start from position 1 to preserve the first token
        min(len(init), batch_size + 1),  # Make sure we don't exceed the input length
        max(1, (len(init) - 1) / max(1, batch_size)),  # Ensure we have a valid step size
        device=grad.device
    ).type(torch.long)

    # Safety check - if new_token_pos is empty or has fewer elements than batch_size
    if len(new_token_pos) < batch_size:
        # If we have too few positions, repeat the last position
        new_token_pos = torch.cat([
            new_token_pos,
            torch.full((batch_size - len(new_token_pos),), new_token_pos[-1] if len(new_token_pos) > 0 else 1,
                       device=grad.device, dtype=torch.long)
        ])

    # Choose a random alternative token for each position
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1,
        torch.randint(0, topk, (batch_size, 1), device=grad.device)
    )

    # Create new candidates by replacing one token in each sequence
    new_control_toks = original_control_toks.scatter_(
        1, new_token_pos.unsqueeze(-1), new_token_val
    )

    return new_control_toks


def step_agg(model_1, model_2, tokenizer_1, tokenizer_2, init, batch_size=1024, topk=5, topk_semanteme=10):
    main_device = model_1.device

    # Make sure init is on the right device
    if not init.is_cuda:
        init = init.to(main_device)

    # Print some debug info
    print(f"Running step_agg with init shape: {init.shape}, batch_size: {batch_size}, topk: {topk}")
    print(f"Init tokens: {init.tolist()}")

    # Compute gradients
    try:
        grad = token_gradients_agg(model_1, model_2, tokenizer_1, tokenizer_2, init)
    except Exception as e:
        print(f"Error in token_gradients_agg: {e}")
        raise

    # Generate candidate token sequences
    try:
        with torch.no_grad():
            control_cand = sample_control_agg(model_1, model_2, tokenizer_1, tokenizer_2, init, grad, batch_size, topk,
                                              topk_semanteme)
    except Exception as e:
        print(f"Error in sample_control_agg: {e}")
        raise

    del grad
    gc.collect()

    # Evaluate candidates
    loss = torch.zeros(batch_size, device=main_device)
    loss_1 = torch.zeros(batch_size, device=main_device)
    loss_2 = torch.zeros(batch_size, device=main_device)
    prob_1 = torch.zeros(batch_size, device=main_device)
    prob_2 = torch.zeros(batch_size, device=main_device)

    eos_token_id_1 = tokenizer_1.eos_token_id
    eos_token_id_2 = tokenizer_2.eos_token_id

    # Get vocabulary sizes for clamping
    vocab_size_1 = model_1.get_input_embeddings().weight.shape[0]
    vocab_size_2 = model_2.get_input_embeddings().weight.shape[0]

    with torch.no_grad():
        for j, cand in enumerate(control_cand):
            try:
                # Ensure tokens are within vocabulary bounds for each model
                valid_cand_1 = torch.clamp(cand, 0, vocab_size_1 - 1)
                valid_cand_2 = torch.clamp(cand, 0, vocab_size_2 - 1)

                # Evaluate on model 1
                full_input_1 = valid_cand_1.unsqueeze(0)
                logits_1 = model_1(full_input_1).logits
                shift_logits_1 = logits_1[..., -1, :].contiguous()
                loss_1[j] = nn.CrossEntropyLoss()(
                    shift_logits_1.view(-1, shift_logits_1.size(-1)),
                    torch.tensor([eos_token_id_1], device=main_device)
                )
                prob_1[j] = torch.nn.functional.softmax(logits_1[0, -1, :], dim=0)[eos_token_id_1]

                # Evaluate on model 2
                full_input_2 = valid_cand_2.unsqueeze(0)
                logits_2 = model_2(full_input_2).logits
                shift_logits_2 = logits_2[..., -1, :].contiguous()
                loss_2[j] = nn.CrossEntropyLoss()(
                    shift_logits_2.view(-1, shift_logits_2.size(-1)),
                    torch.tensor([eos_token_id_2], device=main_device)
                )
                prob_2[j] = torch.nn.functional.softmax(logits_2[0, -1, :], dim=0)[eos_token_id_2]

                # Combined loss
                loss[j] = (loss_1[j] + loss_2[j]) / 2

            except RuntimeError as e:
                print(f"Error evaluating candidate {j}: {e}")
                loss[j] = float('inf')
                loss_1[j] = float('inf')
                loss_2[j] = float('inf')
                prob_1[j] = 0.0
                prob_2[j] = 0.0

        # Find best candidate
        valid_indices = ~torch.isinf(loss)
        if torch.any(valid_indices):
            min_idx = loss[valid_indices].argmin()
            min_idx = torch.nonzero(valid_indices)[min_idx]
        else:
            print("Warning: All candidates produced errors. Returning original init.")
            return init, float('inf'), float('inf'), float('inf'), 0.0, 0.0

        next_control = control_cand[min_idx]
        cand_loss = loss[min_idx]
        cand_loss_1 = loss_1[min_idx]
        cand_loss_2 = loss_2[min_idx]
        cand_prob_1 = prob_1[min_idx]
        cand_prob_2 = prob_2[min_idx]

    del loss, loss_1, loss_2, prob_1, prob_2
    gc.collect()

    return next_control, cand_loss, cand_loss_1, cand_loss_2, cand_prob_1, cand_prob_2