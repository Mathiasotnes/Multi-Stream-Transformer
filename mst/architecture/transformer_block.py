##############################################################################
## transformer_block.py                                                     ##
## -----------------------                                                  ##
## Transformer block implementation.                                        ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

import torch.nn as nn
from torch import Tensor
from .multi_head_attention import MultiHeadAttention

class TransformerBlock( nn.Module ):
    """
    Transformer block consisting of: 
        1. Multi-Head Attention
        2. Add & Norm
        3. Feed Forward (Multi-Layer Perceptron)
        4. Add & Norm
    """
    def __init__( 
            self,
            num_heads:  int,
            d_model:    int,
            hidden_dim: int,
            masked:     bool=False,
            dropout:    float=0.1, 
        ) -> None:
        """
        Args:
            num_heads (int):                Number of attention heads.
            d_model (int):                  Dimensionality of the model (Embedding dimension).
            hidden_dim (int):               Hidden layer dimensionality (MLP).
            masked (bool, optional):        Whether to use causal masked attention or not. Defaults to False.
            dropout (float, optional):      Dropout rate. Defaults to 0.1.
        """
        super(TransformerBlock, self).__init__()
        self.attn   = MultiHeadAttention(d_model, num_heads, masked=masked)
        self.ln1    = nn.LayerNorm(d_model)
        self.mlp    = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.ln2    = nn.LayerNorm(d_model)
        self.do     = nn.Dropout(dropout)
    
    def forward( self, x: Tensor ) -> Tensor:
        """
        Forward pass through the transformer block. Results from mlp and multi-head attention
        are added to the input tensor to create residual connections.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.do(self.mlp(self.ln2(x)))
        return x
    