##############################################################################
## open_web_text.py                                                         ##
## ----------------                                                         ##
## Dataloader for the OpenWebText dataset. Which is one of the datasets     ##
## used in the RoBERTa model:                                               ##
## - https://arxiv.org/pdf/1907.11692                                       ##
## - https://huggingface.co/datasets/Skylion007/openwebtext                 ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from typing import Any, Iterator

class OpenWebTextDataset( IterableDataset ):
    """
    Iterable Dataset for the OpenWebText dataset, used in the RoBERTa model.
    This dataset streams data without downloading the entire dataset.

    Args:
        tokenizer (Any):               Tokenizer to tokenize the text data.
        max_seq_length (int):          Maximum sequence length for tokenization.
    """
    def __init__( self, tokenizer: Any, max_seq_length: int ) -> None:
        super(OpenWebTextDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __iter__( self ) -> Iterator[torch.Tensor]:
        """
        Iterate over the dataset and yield tokenized samples.

        Yields:
            torch.Tensor: Tokenized input_ids tensor.
        """
        # Load the dataset in streaming mode
        dataset = load_dataset('openwebtext', split='train', streaming=True, trust_remote_code=True)

        for sample in dataset:
            text = sample['text']
            # Tokenize the text
            tokens = self.tokenizer(
                text, 
                truncation=True, 
                max_length=self.max_seq_length, 
                return_tensors='pt'
            )
            # Yield the input_ids tensor
            yield tokens['input_ids'].squeeze(0)

def get_owt_dataloader( 
        tokenizer: Any, 
        max_seq_length: int, 
        batch_size: int, 
        collate_fn: callable, 
        split: str='train' 
    ) -> DataLoader:
    """
    Get a DataLoader for the OpenWebText dataset.

    Args:
        tokenizer (Any):           Tokenizer to tokenize the text data.
        max_seq_length (int):      Maximum sequence length for tokenization.
        batch_size (int):          Batch size.
        split (str, optional):     Dataset split, defaults to 'train'.

    Returns:
        DataLoader:                DataLoader for the dataset.
    """
    if split == 'train':
        dataset = OpenWebTextDataset(tokenizer=tokenizer, max_seq_length=max_seq_length)
    else:
        dataset = None
        raise ValueError(f"Unknown split '{split}' for OpenWebText dataset.")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn
    )
    return dataloader
