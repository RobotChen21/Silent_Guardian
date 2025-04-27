import gc
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sentence_transformers.util import (semantic_search,
                                        dot_score,
                                        normalize_embeddings)

def token_gradients(model, tokenizer, init):
    embed_weights = model.get_input_embeddings().weight 
    one_hot = torch.zeros(
        init.shape[0],# len_of_token
        embed_weights.shape[0],# size_of_vocab
        device=model.device,
        dtype=embed_weights.dtype
    )
    #print(embed_weights.dtype)
    one_hot.scatter_(
        1, 
        init.unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()#finish create one hot
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)#batch,len,weight
    logits = model(inputs_embeds=input_embeds).logits
    shift_logits = logits[..., -1, :].contiguous()
    eos_token_id = tokenizer.eos_token_id
    loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), torch.tensor([eos_token_id]).cuda())  ### token as added one, end token.
    
    loss.backward()
    return one_hot.grad.clone()#computate grad only

def sample_control(model, init, grad, batch_size, topk=5, topk_semanteme=10):
    curr_embeds = model.get_input_embeddings()(init.unsqueeze(0))
    curr_embeds = torch.nn.functional.normalize(curr_embeds,dim=2)               # queries

    embedding_matrix = model.get_input_embeddings().weight
    embedding_matrix = normalize_embeddings(embedding_matrix)      # corpus

    top_indices=torch.zeros(grad.shape[0],topk,dtype=init.dtype).to("cuda")
    for i in range(init.shape[0]):
        similar=[]    
        temp=[]                 
        query=curr_embeds[0][i]                 #query
        corpus=embedding_matrix                 #corpus
        hits = semantic_search(query, corpus, 
                            top_k=topk_semanteme+1,
                            score_function=dot_score)
        for hit in hits:
            # don't choose same token as before
            for dic in hit[1:]:
                temp.append(((grad)[i][dic["corpus_id"]],dic["corpus_id"]))
        temp.sort()
        # in topksemanteme choose topk to use
        for j in range(topk): 
            similar.append(temp[j][1])
        top_indices[i]=torch.tensor(similar)

    """
    [[1,5,9,...],
    [1,5,9,...],
    [1,5,9,...],
    ...]
    """
    #thinking: if make adv directly,[0,0,0,1,0,0,...] will - grda*learning_rate,trending to another one hot vector.

    #[!,!,!,...]
    original_control_toks = init.repeat(batch_size, 1)
    """
    [[!,!,!,...],
    [!,!,!,...],
    [!,!,!,...],
    ...]batch_size
    ! is index of vocab
    """
    if batch_size > init.shape[0]:
        new_token_pos = torch.arange(      ##don't touch <s>
            1, 
            len(init), 
            (len(init)-1) / batch_size,      ## this place has to change as well
            device=grad.device
        ).type(torch.int64)
    else:
        new_token_pos =  (torch.randperm(init.shape[0]-1)[0:batch_size] + 1).to("cuda")  ##don't touch <s>
        
    """
    [...,...,...]
    """

    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1),device=grad.device)
    )
    """
    [[top_indices],
    [1],
    [1],
    [1],
    ...]
    """
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
    """
    self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    self[i][index[i][j]] = src[i][j]  # if dim == 1
    self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

    every control[!,!,!,...] be changed one position.
    """
    return new_control_toks

def step(model, init, tokenizer, batch_size=1024, topk=5, topk_semanteme=10):

    main_device = model.device
    # Aggregate gradients.universal needs to add all gradient.
    grad = token_gradients(model, tokenizer, init)

    with torch.no_grad():
        control_cand = sample_control(model, init, grad, batch_size, topk, topk_semanteme)
    del grad ; gc.collect()

    # Search
    loss = torch.zeros( batch_size).to(main_device)
    prob = torch.zeros( batch_size).to(main_device)
    eos_token_id = tokenizer.eos_token_id
    with torch.no_grad():
        for j, cand in enumerate(control_cand):
            full_input = cand.unsqueeze(0)
            logits = model(full_input).logits
            shift_logits = logits[..., -1, :].contiguous()
            #LLama模型的endtoken是[2]
            # loss[j] = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), torch.tensor([2]).cuda())  ###token as added one,end token
            # prob[j] = torch.nn.functional.softmax(logits[0,-1,:],dim=0)[2] #end token
            # Deepseek的end token是EOS token ID: 100001
            loss[j] = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), torch.tensor([eos_token_id]).cuda())  ###token as added one,end token
            prob[j] = torch.nn.functional.softmax(logits[0,-1,:],dim=0)[eos_token_id] #end token
        min_idx = loss.argmin()
        next_control, cand_loss, cand_prob= control_cand[min_idx], loss[min_idx], prob[min_idx]

    del  loss,prob ; gc.collect()
    return next_control, cand_loss, cand_prob


