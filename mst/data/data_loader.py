##############################################################################
## data_loader.py                                                           ##
## ----------------                                                         ##
## General DataLoader setup that takes dataset and tokenizer names as input ##
## strings, sets up the appropriate tokenizer and dataset, and returns a    ##
## DataLoader.                                                              ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

import torch
from torch.utils.data import DataLoader
from typing import Any, List
import torch.nn.utils.rnn as rnn_utils
from transformers import AutoTokenizer
from .open_web_text import OpenWebTextDataset

def get_tokenizer( tokenizer_name: str ) -> Any:
    """
    Initialize and return a tokenizer based on the tokenizer name.
    
    Args:
        tokenizer_name (str): Name of the tokenizer.
    
    Returns:
        Any: The tokenizer object.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    except Exception as e:
        raise ValueError(f"Unknown tokenizer '{tokenizer_name}': {e}")
    return tokenizer

def get_dataset( dataset_name: str, tokenizer: Any, max_seq_length: int ) -> torch.utils.data.Dataset:
    """
    Initialize and return a dataset based on the dataset name.
    
    Args:
        dataset_name (str):     Name of the dataset.
        tokenizer (Any):        Tokenizer to use.
        max_seq_length (int):   Maximum sequence length.
    
    Returns:
        torch.utils.data.Dataset: The dataset object.
    """
    if dataset_name.lower() == 'openwebtext':
        return OpenWebTextDataset(tokenizer=tokenizer, max_seq_length=max_seq_length)
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

def collate_fn( batch: List[torch.Tensor], tokenizer: Any ) -> torch.Tensor:
    """
    Collate function to pad variable length sequences.

    Args:
        batch (List[torch.Tensor]): List of tokenized input_ids tensors.

    Returns:
        torch.Tensor: Padded input_ids tensor.
    """
    return rnn_utils.pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)

def get_dataloader( 
    dataset_name: str, 
    tokenizer_name: str, 
    max_seq_length: int, 
    batch_size: int 
) -> DataLoader:
    """
    Get a DataLoader for the specified dataset and tokenizer.
    
    Args:
        dataset_name (str):   Name of the dataset.
        tokenizer_name (str): Name of the tokenizer.
        max_seq_length (int): Maximum sequence length for tokenization.
        batch_size (int):     Batch size.
    
    Returns:
        DataLoader:           DataLoader for the dataset.
    """
    tokenizer = get_tokenizer(tokenizer_name)
    dataset = get_dataset(dataset_name, tokenizer, max_seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    return dataloader
