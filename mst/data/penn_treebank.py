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

class PennTreebankTrainDataset( IterableDataset ):
    """
    Iterable Dataset for the PennTreebankTrainDataset dataset.

    Args:
        tokenizer (Any):               Tokenizer to tokenize the text data.
        max_seq_length (int):          Maximum sequence length for tokenization.
    """
    def __init__( self, tokenizer: Any, max_seq_length: int ) -> None:
        super(PennTreebankTrainDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __iter__( self ) -> Iterator[torch.Tensor]:
        """
        Iterate over the dataset and yield tokenized samples.

        Yields:
            torch.Tensor: Tokenized input_ids tensor.
        """
        # Load the dataset in streaming mode
        dataset = load_dataset('ptb_text_only', split='train', streaming=True, trust_remote_code=True)

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

class PennTreebankValDataset( IterableDataset ):
    """
    Iterable Dataset for the PennTreebankValDataset dataset.

    Args:
        tokenizer (Any):               Tokenizer to tokenize the text data.
        max_seq_length (int):          Maximum sequence length for tokenization.
    """
    def __init__( self, tokenizer: Any, max_seq_length: int ) -> None:
        super(PennTreebankValDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __iter__( self ) -> Iterator[torch.Tensor]:
        """
        Iterate over the dataset and yield tokenized samples.

        Yields:
            torch.Tensor: Tokenized input_ids tensor.
        """
        # Load the dataset in streaming mode
        dataset = load_dataset('ptb_text_only', split='validation', streaming=True, trust_remote_code=True)

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

def count_tokens(dataset):
    token_count = 0
    for sample in dataset:
        token_count += sample.size(0)
    return token_count

def get_ptb_dataloaders( tokenizer: Any, max_seq_length: int, batch_size: int, collate_fn: callable ) -> DataLoader:
    """
    Get a DataLoader for the PennTreebank dataset.

    Args:
        tokenizer (Any):           Tokenizer to tokenize the text data.
        max_seq_length (int):      Maximum sequence length for tokenization.
        batch_size (int):          Batch size.

    Returns:
        Tuple[DataLoader]:         DataLoaders for the dataset.
    """
    train_dataset = PennTreebankTrainDataset(tokenizer=tokenizer, max_seq_length=max_seq_length)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn
    )
    val_dataset = PennTreebankValDataset(tokenizer=tokenizer, max_seq_length=max_seq_length)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn
    )
    
    # print(f"\r\nPenn Treebank Dataset:")
    # print(f"{'='*40}")
    # print(f"First 10 sentences in training set:")
    # for i, sample in enumerate(train_dataset):
    #     if i == 10:
    #         break
    #     print(f" * {i+1:<2} | {tokenizer.decode(sample)}")
    
    # print(f"{'-'*40}")
    # print(f"First 10 sentences in validation set:")
    # for i, sample in enumerate(val_dataset):
    #     if i == 10:
    #         break
    #     print(f" * {i+1:<2} | {tokenizer.decode(sample)}")
    
    # train_tokens = count_tokens(train_dataset) # 1094393 with gpt2 tokenizer
    # val_tokens = count_tokens(val_dataset) # 86474 with gpt2 tokenizer
    
    # print(f"{'-'*40}")
    # print(f"{'':<10} | {'Train':<10} | {'Validation':<10}")
    # print(f"{'-'*40}")
    # print(f"{' * Tokens:':<10} | {train_tokens:<10} | {val_tokens:<10}")
    # print(f"{'='*40}")
    
    return train_dataloader, val_dataloader