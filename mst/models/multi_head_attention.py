##############################################################################
## multi_head_attention.py                                                  ##
## -----------------------                                                  ##
## Multi-head attention implementation. Inspired by:                        ##
## - https://github.com/karpathy/minGPT/blob/master/mingpt/model.py         ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MultiHeadAttention( nn.Module ):
    """
    Multi-head attention mechanism.
    
    Args:
        d_model (int):                  Dimensionality of the model (Embedding dimension).
        num_heads (int):                Number of attention heads.
        context_window (int):           Size of the context window. Defaults to d_model.
        masked (bool):                  Whether to use causal masked attention or not.
    """
    def __init__( 
            self, 
            d_model: int, 
            num_heads: int, 
            context_window: int=None, 
            masked: bool=False,
        ) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.nh         = num_heads
        self.d_k        = d_model // self.nh # Embedding dimension for each head
        self.d_model    = d_model
        self.ctx_d      = context_window if context_window is not None else d_model
        self.masked     = masked

        self.attn = nn.Linear(d_model, 3 * d_model) 
        self.W_o = nn.Linear(d_model, d_model)
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(self.ctx_d, self.ctx_d)).view(1, 1, self.ctx_d, self.ctx_d) # (1, 1, ctx_d, ctx_d) - Causal mask
        )
        
    def forward( self, x: Tensor ) -> Tensor:
        B, T, C = x.size() # (batch_size, seq_length, d_model)

        # Calculate queries, keys and values for all heads in a batch, and move head dimension to the front
        Q, K, V = self.attn(x).split(self.d_model, dim=2)   # (B, T, 3*d_model) -> 3 * (B, T, d_model)
        Q = Q.view(B, T, self.nh, self.d_k).transpose(1, 2) # (B, T, nh, d_k)   -> (B, nh, T, d_k)
        K = K.view(B, T, self.nh, self.d_k).transpose(1, 2) # (B, T, nh, d_k)   -> (B, nh, T, d_k)
        V = V.view(B, T, self.nh, self.d_k).transpose(1, 2) # (B, T, nh, d_k)   -> (B, nh, T, d_k)

        # Scaled dot-product attention
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if self.masked:
            attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        self.attn_weights = attn
        y = attn @ V
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.W_o(y)
        return y
    