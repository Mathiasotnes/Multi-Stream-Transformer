##############################################################################
## multi_stream_transformer.py                                              ##
## ---------------------------                                              ##
## Multi-stream transformer model implementation. This model processes the  ##
## input data through multiple parallell residual streams before combining  ##
## them in downstream transformer blocks.                                   ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .pos_encoding import PositionalEncoding
from .transformer_block import TransformerBlock


class MultiStreamTransformer( nn.Module ):
    """
    Multi-stream transformer model. Consists of:
        1. Embedding layer
        2. Positional encoding
        3. Multi-stream transformer blocks
        4. Stream fusion layer
        5. Transformer blocks
    """
    def __init__(
        self,
        vocab_size: int, 
        d_model: int, 
        num_heads: int, 
        hidden_dim: int, 
        num_streams: int,
        blocks_per_stream: int,
        num_blocks: int,
        dropout: float=0.1,
        fusion_method: str = 'average',
        echo_specs: bool=True
    ) -> None:
        """
        Args:
            vocab_size (int):               Size of the vocabulary.
            d_model (int):                  Dimensionality of the model (Embedding dimension).
            num_heads (int):                Number of attention heads.
            hidden_dim (int):               Hidden layer dimensionality (MLP).
            num_streams (int):              Number of parallel streams.
            blocks_per_stream (int):        Number of transformer blocks per stream.
            num_blocks (int):               Number of transformer blocks.
            dropout (float, optional):      Dropout rate. Defaults to 0.1.
            fusion_method (str, optional):  Method for fusing the streams. Defaults to 'average'.
            echo_spechs (bool, optional):   Whether to print the model specifications. Defaults to True.
        """
        super(MultiStreamTransformer, self).__init__()
        
        assert fusion_method in ['average'], "Invalid fusion method. Must be one of ['average']"
        
        self.vocab_size         = vocab_size
        self.d_model            = d_model
        self.num_heads          = num_heads
        self.hidden_dim         = hidden_dim
        self.num_streams        = num_streams
        self.blocks_per_stream  = blocks_per_stream
        self.num_blocks         = num_blocks
        self.dropout            = dropout
        self.fusion_method      = fusion_method
        
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        
        self.streams = nn.ModuleList([
            nn.Sequential(*[
                TransformerBlock(
                    num_heads=num_heads, 
                    d_model=d_model, 
                    hidden_dim=hidden_dim, 
                    masked=True, 
                    dropout=dropout
                ) for _ in range(blocks_per_stream)
            ]) for _ in range(num_streams)
        ])
        
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                num_heads=num_heads, 
                d_model=d_model, 
                hidden_dim=hidden_dim, 
                masked=True, 
                dropout=dropout
            ) for _ in range(num_blocks)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        if echo_specs: print(self)
        
    def __repr__( self ) -> str:
        """
        Returns a string representation of the model specifications.
        """
        total_params = sum(p.numel() for p in self.parameters())
        
        model_str = f"\n\rMulti-Stream Transformer Model Specifications:\n"
        model_str += f"{'='*40}\n"
        model_str += f"Vocabulary Size:          {self.vocab_size}\n"
        model_str += f"Embedding Dimension:      {self.d_model}\n"
        model_str += f"Number of Heads:          {self.num_heads}\n"
        model_str += f"Number of Blocks:         {self.num_blocks}\n"
        model_str += f"Number of Streams:        {self.num_streams}\n"
        model_str += f"Blocks per Stream:        {self.blocks_per_stream}\n"
        model_str += f"Hidden Dimension (MLP):   {self.hidden_dim}\n"
        model_str += f"Dropout Rate:             {self.dropout}\n"
        model_str += f"Total Parameters:         {total_params}\n"
        model_str += f"{'='*40}\n"
        model_str += f"Parameter count:\n"
        
        # Components and their parameter counts
        components = [
            ('Embedding Layer:     ', self.emb),
            ('Positional Encoding: ', self.pos_enc),
            ('Linear Head:         ', self.lm_head),
            ('Layer Norm:          ', self.ln_f),
        ]
        
        # Add Streams
        for i, stream in enumerate(self.streams):
            for j, block in enumerate(stream):
                components.append((f'Stream {i+1} Block {j+1}:    ', block))
                
        # Add Transformer Blocks
        for i, block in enumerate(self.blocks):
            components.append((f'Transformer Block {i+1}: ', block))
        
        # Calculate and append parameter counts for each component
        for name, module in components:
            num_params = sum(p.numel() for p in module.parameters())
            model_str += f"  * {name} {num_params}\n"
        
        model_str += f"{'='*40}"
        return model_str

    def forward( self, x: Tensor, targets: Tensor=None ) -> Tensor:
        """
        Forward pass through the multi-stream transformer.

        Args:
            x (Tensor):                     Input tensor.
            targets (Tensor, optional):     Target tensor. Defaults to None.

        Returns:
            logits: (Tensor)                Logits tensor.
            loss: (Tensor)                  Loss value. Defaults to None.
        """
        x = self.emb(x)
        x = self.pos_enc(x)
        
        # Process the streams
        stream_outputs = []
        for stream in self.streams:
            stream_output = x
            for block in stream:
                stream_output = block(stream_output)
            stream_outputs.append(stream_output)
        
        # Fuse the streams
        if self.fusion_method == 'average':
            x = torch.stack(stream_outputs).mean(dim=0)
        
        # Process the transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                reduction='mean'
            )
        
        return logits, loss
