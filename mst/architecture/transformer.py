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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .pos_encoding import PositionalEncoding
from .transformer_block import TransformerBlock

class Transformer( nn.Module ):
    """
    Transformer model. Consists of:
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
        super(Transformer, self).__init__()
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
        model_str = f"\n\rTransformer Model Specifications:\n"
        model_str += f"{'='*40}\n"
        model_str += f"Vocabulary Size:          {self.vocab_size}\n"
        model_str += f"Embedding Dimension:      {self.d_model}\n"
        model_str += f"Number of Heads:          {self.num_heads}\n"
        model_str += f"Number of Blocks:         {self.num_blocks}\n"
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

        # Add Transformer Blocks
        for i, block in enumerate(self.blocks):
            components.append((f'Transformer Block {i+1}: ', block))

        # Calculate and append parameter counts for each component
        for name, module in components:
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            model_str += f"  * {name} {num_params}\n"

        model_str += f"{'='*40}"
        return model_str
    
    def forward( self, x: Tensor, targets: Tensor=None ) -> Tensor:
        """
        Forward pass through the transformer.

        Args:
            x (Tensor):                     Input tensor. (batch_size, seq_length)
            targets (Tensor, optional):     Target tensor. Defaults to None.

        Returns:
            logits (Tensor):                Logits tensor. (batch_size, seq_length, vocab_size)
            loss (Tensor):                  Loss value. Defaults to None.
            
        """
        x = self.emb(x)
        x = self.pos_enc(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # targets: (batch_size, seq_length)
            # logits:  (batch_size, seq_length, vocab_size)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                reduction='mean'
            )

        return logits, loss
    
    def generate(
        self,
        input_ids: Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = 1.0,
    ) -> Tensor:
        """
        Generate text given an input prompt.

        Args:
            input_ids (Tensor):         Input tensor of shape (batch_size, seq_length)
            max_length (int):           Maximum length of the generated sequence
            temperature (float):        Sampling temperature
            top_k (int):                If specified, only sample from the top_k most probable tokens
            top_p (float):              If specified, only sample from the top_p cumulative probability
            repetition_penalty (float): Penalty for repeated tokens

        Returns:
            output_ids (Tensor): Generated sequence of token ids
        """
        device = input_ids.device
        output_ids = input_ids.clone()
        past_tokens = set()

        for _ in range(max_length):
            logits, _ = self.forward(output_ids)
            logits = logits[:, -1, :] / temperature  # (batch_size, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in past_tokens:
                    logits[:, token_id] /= repetition_penalty

            # Apply top_k and top_p filtering
            filtered_logits = self.filter_logits(logits, top_k=top_k, top_p=top_p)

            # Sample from the filtered distribution
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Update output_ids and past_tokens
            output_ids = torch.cat([output_ids, next_token], dim=1)
            past_tokens.add(next_token.squeeze().tolist())

            # Break if EOS token is generated (assuming EOS token ID is known)
            # if next_token.item() == eos_token_id:
            #     break

        return output_ids

    def filter_logits(self, logits, top_k=None, top_p=None, filter_value=-float('Inf')):
        """
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

        Args:
            logits (Tensor):        Logits distribution shape (batch_size, vocab_size)
            top_k (int):            Keep only top k tokens with highest probability
            top_p (float):          Keep the top tokens with cumulative probability >= top_p
            filter_value (float):   The value to replace filtered logits with

        Returns:
            logits (Tensor): Filtered logits
        """
        if top_k is not None and top_k > 0:
            top_k = min(max(top_k, 1), logits.size(-1))  # Safety check
            # Remove tokens with probability less than the kth highest
            topk_vals, _ = torch.topk(logits, top_k)
            min_topk_vals = topk_vals[:, -1].unsqueeze(-1)
            indices_to_remove = logits < min_topk_vals
            logits = logits.masked_fill(indices_to_remove, filter_value)

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p

            # Shift the indices to the right to include at least one token
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0

            # Create a mask tensor in sorted order
            mask = sorted_indices_to_remove

            # Unsort the mask to the original order
            logits_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            logits_to_remove.scatter_(dim=-1, index=sorted_indices, src=mask)

            # Apply the mask
            logits = logits.masked_fill(logits_to_remove, filter_value)

        return logits
