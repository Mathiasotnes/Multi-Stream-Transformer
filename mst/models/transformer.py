##############################################################################
## transformer.py                                                           ##
## ---------------                                                          ##
## Transformer model implementation. This is used as a baseline model for   ##
## comparison to the multi-stream transformer model. This is a decoder-only ##
## transformer model optimized for text-generation                          ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .pos_encoding import PositionalEncoding
from .transformer_block import TransformerBlock

class Decoder( nn.Module ):
    """
    Decoder model. Consists of:
        1. Embedding layer
        2. Positional encoding
        3. Transformer blocks (masked)
    """
    def __init__( 
            self, 
            vocab_size: int, 
            d_model: int, 
            num_heads: int, 
            hidden_dim: int, 
            num_blocks: int, 
            dropout: float=0.1,
            echo_specs: bool=True
        ) -> None:
        """
        Args:
            vocab_size (int):               Size of the vocabulary.
            d_model (int):                  Dimensionality of the model (Embedding dimension).
            num_heads (int):                Number of attention heads.
            hidden_dim (int):               Hidden layer dimensionality (MLP).
            num_blocks (int):               Number of transformer blocks.
            dropout (float, optional):      Dropout rate. Defaults to 0.1.
            echo_spechs (bool, optional):   Whether to print the model specifications. Defaults to True.
        """
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(num_heads=num_heads, d_model=d_model, hidden_dim=hidden_dim, masked=True, dropout=dropout) for _ in range(num_blocks)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        if echo_specs: print(self)
    
    def __repr__( self ) -> str:
        """
        Returns a string representation of the model specifications.
        """
        # Calculate total trainable parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Build the string
        model_str = f"\n\rDecoder Model Specifications:\n"
        model_str += f"{'='*40}\n"
        model_str += f"Vocabulary Size:          {self.vocab_size}\n"
        model_str += f"Embedding Dimension:      {self.d_model}\n"
        model_str += f"Number of Heads:          {self.num_heads}\n"
        model_str += f"Number of Blocks:         {self.num_blocks}\n"
        model_str += f"Hidden Dimension (MLP):   {self.hidden_dim}\n"
        model_str += f"Dropout Rate:             {self.dropout}\n"
        model_str += f"Total Parameters:         {total_params}\n"
        model_str += f"{'='*40}\n"
        model_str += f"Trainable Parameters per Component:\n"

        # Components and their parameter counts
        components = [
            ('Embedding Layer:    ', self.emb),
            ('Positional Encoding:', self.pos_enc),
            ('Linear Head:        ', self.lm_head),
            ('Layer Norm:         ', self.ln_f),
        ]

        # Add Transformer Blocks
        for i, block in enumerate(self.blocks):
            components.append((f'Transformer Block {i+1}:', block))

        # Calculate and append parameter counts for each component
        for name, module in components:
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            model_str += f"  * {name} {num_params}\n"

        model_str += f"{'='*40}\n"
        return model_str
    
    def forward( self, x: Tensor, targets: Tensor=None ) -> Tensor:
        """
        Forward pass through the decoder.

        Args:
            x (Tensor):                     Input tensor. (batch_size, seq_length)
            targets (Tensor, optional):     Target tensor. Defaults to None.

        Returns:
            x (Tensor):                     Decoded representation. (batch_size, seq_length, d_model)
            loss (Tensor):                  Loss value. Defaults to None.
            
        """
        x = self.emb(x)
        x = self.pos_enc(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return x, loss

