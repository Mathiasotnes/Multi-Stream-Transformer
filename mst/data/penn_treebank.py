##############################################################################
## penn_treebank.py                                                         ##
## ----------------                                                         ##
## Dataloader for the Penn Treebank dataset. Which is widely used for       ##
## benchmarking language modelling models. It consists of 912 344 tokens    ##
## (38 219 sentences) in the training set and 131 768 (5 527 sentences)     ##
## in the validation set.                                                   ##
##                                                                          ##
## - https://paperswithcode.com/dataset/penn-treebank                       ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from typing import Any, Iterator

class PennTreebankDataset( IterableDataset ):
    """
    Iterable Dataset for the PennTreebankDataset dataset.

    Args:
        tokenizer (Any):               Tokenizer to tokenize the text data.
        max_seq_length (int):          Maximum sequence length for tokenization.
    """
    def __init__( self, tokenizer: Any, split: str, max_seq_length: int ) -> None:
        super(PennTreebankDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.split = split

    def __iter__( self ) -> Iterator[torch.Tensor]:
        """
        Iterate over the dataset and yield tokenized samples.

        Yields:
            torch.Tensor: Tokenized input_ids tensor.
        """
        # Load the dataset in streaming mode
        dataset = load_dataset('ptb_text_only', split=self.split, streaming=True, trust_remote_code=True)

        for sample in dataset:
            text = sample['sentence']
            # Tokenize the text
            tokens = self.tokenizer(
                text, 
                truncation=True, 
                max_length=self.max_seq_length, 
                return_tensors='pt'
            )
            # Yield the input_ids tensor
            yield tokens['input_ids'].squeeze(0)

def get_ptb_dataloader( tokenizer: Any, max_seq_length: int, batch_size: int, collate_fn: callable, split: str='train' ) -> DataLoader:
    """
    Get a DataLoader for the PennTreebank dataset.

    Args:
        tokenizer (Any):           Tokenizer to tokenize the text data.
        max_seq_length (int):      Maximum sequence length for tokenization.
        batch_size (int):          Batch size.
        split (str):               Split of the dataset to use.

    Returns:
        DataLoader:                DataLoader for the dataset.
    """
    dataset = PennTreebankDataset(tokenizer, split, max_seq_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn
    )
    return dataloader
    