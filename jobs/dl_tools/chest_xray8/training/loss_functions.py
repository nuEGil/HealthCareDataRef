import torch.nn as nn
import torch

def NTXEntLoss(z, n_samps, device):
    # follow https://arxiv.org/abs/2002.05709
    # set xk including a positive pair of example xj and xk
    tau = 0.5 # temperature param
    cos = nn.CosineSimilarity(dim=0, eps=1e-6) # compute along dim 
    sij = torch.zeros((2*n_samps, 2*n_samps),
        dtype=torch.float32, device=device)
    # print('sij shape ', sij.shape)
    for i in range(2*n_samps):
        for j in range(2*n_samps):
            sij[i, j] = cos(z[i,...], z[j,...]) # reference makes this (features,) not (batch, features)
    
    numer = torch.exp(sij/tau)
    
    mask = (1-torch.eye(2*n_samps, device=device))
    denom = mask * torch.exp(sij/tau)
    denom = denom.sum(dim = 1, keepdim=True)
    
    # lij 
    lij = -torch.log(numer /denom.squeeze())

    loss = 0
    for k in range(1, n_samps+1):
        a = (2*k-1) -1
        b = (2*k) -1
        loss+= (lij[a, b] + lij[b, a])
    loss = loss / (2*n_samps)
    # print('batch loss : ', loss)
    return loss