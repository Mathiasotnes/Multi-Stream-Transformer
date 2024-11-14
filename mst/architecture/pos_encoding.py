##############################################################################
## pos_encoding.py                                                          ##
## ----------------                                                         ##
## Positional encoding implementation. Utilizing absolute positional        ##
## encoding.                                                                ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

import torch
import torch.nn as nn
from torch import Tensor

class PositionalEncoding( nn.Module ):
    """
    Absolute positional encoding module using learnable embeddings.
    """
    def __init__( self, d_model: int, dropout: float = 0.1, max_length: int = 5000 ) -> None:
        """
        Args:
            d_model (int):                  Dimension of embeddings.
            dropout (float, optional):      Dropout rate, defaults to 0.1.
            max_length (int, optional):     Max sequence length, defaults to 5000.
        """
        super().__init__()
        self.do = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_length, d_model)

    def forward( self, x: Tensor ) -> Tensor:
        """
        Args:
            x (Tensor):     Input tensor of shape (batch_size, seq_length, d_model).
        
        Returns:
            Tensor:         The input tensor with positional embeddings added.
        """
        B, T, C = x.size() # (batch_size, seq_length, d_model)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T) # (0, 1, ..., seq_length-1) for each batch
        pe = self.pe(pos)
        x = x + pe
        x = self.do(x)
        return x
     