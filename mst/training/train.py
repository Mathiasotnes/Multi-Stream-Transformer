##############################################################################
## train.py                                                                 ##
## --------                                                                 ##
## Training setup for language models.                                      ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

from torch.utils.tensorboard import SummaryWriter
from ptflops import get_model_complexity_info
from typing import Any
from torch.utils.data import DataLoader
from itertools import cycle
from datetime import datetime

import torch
import math
import torch.optim as optim
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def estimate_flops(model, sequence_length):
    # Exclude batch size since ptflops adds batch_size=1 automatically
    input_shape = (sequence_length,)  # Sequence length without batch size

    # Define an input_constructor to create the appropriate input tensor
    def input_constructor(input_res):
        seq_len = input_res[0]
        # Create a tensor of zeros (token indices), dtype should be torch.long
        input_ids = torch.zeros((1, seq_len), dtype=torch.long).to(device)
        return input_ids  # Adjusted to return tensor directly if model accepts positional args

    try:
        macs, params = get_model_complexity_info(
            model,
            input_res=input_shape,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
            input_constructor=input_constructor,
            custom_modules_hooks={},
            backend='pytorch',
        )
        flops = 2 * macs
        print(f"MACs: {macs}, Params: {params}, FLOPs: {flops}")
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        flops = None
    return flops


def train_model( model: Any, training_config: dict, train_dataloader: DataLoader, val_dataloader: DataLoader=None ) -> Any:
    
    # Extract training configurations
    lr                              = training_config["learning_rate"]
    eval_interval_tokens            = training_config["eval_interval_tokens"]
    eval_tokens                     = training_config["eval_tokens"]
    tokens_to_train                 = training_config["tokens"]
    model_name                      = training_config["model_name"]
    checkpoint_interval_tokens      = training_config["checkpoint_interval_tokens"]
    checkpoint_dir                  = training_config["checkpoint_dir"]
    log_dir                         = training_config["log_dir"]
    
    # Initialize training parameters
    optimizer           = optim.Adam(model.parameters(), lr=lr)
    train_dataloader    = cycle(train_dataloader)
    tokens_trained      = 0
    perplexities        = []
    next_checkpoint     = checkpoint_interval_tokens
    next_eval_tokens    = eval_interval_tokens
    cumulative_loss     = 0.0
    cumulative_batches  = 0
    start_time          = time.time()
    last_time           = start_time
    
    if val_dataloader is None:
        val_dataloader = train_dataloader

    # Checkpointing
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{model_name}.pt')

    # Logging
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(log_dir, run_id)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Estimate FLOPs
    batch = next(iter(train_dataloader))
    batch_size, seq_length = batch.shape
    input_shape = seq_length
    flops_per_batch = estimate_flops(model, input_shape)
    total_flops = 0

    # Print header information
    print(f"Run ID: {run_id}")
    print(f"Total tokens to train: {tokens_to_train}\n")
    print(f"Model: {model_name}")
    print(f"Using device: {device}\n")
    # Print table header
    print(f"{'Tokens Trained':<15} {'Tokens/Sec':<12} {'FLOPs/Sec':<12} {'Loss':<10} {'Perplexity':<12} {'ETA':<10}")
    print('-' * 85)

    while tokens_trained < tokens_to_train:
        batch_start_time = time.time()

        # Get next batch
        batch = next(train_dataloader)
        xb = batch[:, :-1].to(device)
        yb = batch[:, 1:].to(device)

        # Training step
        optimizer.zero_grad()
        logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

        batch_time = time.time() - batch_start_time

        # Update counters
        batch_tokens = xb.numel()
        tokens_trained += batch_tokens
        cumulative_loss += loss.item()
        cumulative_batches += 1
        total_flops += flops_per_batch

        # Calculate metrics
        avg_loss = cumulative_loss / cumulative_batches
        current_time = time.time()
        total_elapsed_time = current_time - start_time
        tokens_per_second = tokens_trained / total_elapsed_time
        flops_per_second = total_flops / total_elapsed_time

        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_loss, tokens_trained)
        writer.add_scalar('Tokens per second', tokens_per_second, tokens_trained)
        writer.add_scalar('FLOPs per second', flops_per_second, tokens_trained)

        # Evaluation
        if tokens_trained >= next_eval_tokens:
            perplexity = compute_perplexity(model, train_dataloader, eval_tokens)
            perplexities.append(perplexity)
            writer.add_scalar('Perplexity/train', perplexity, tokens_trained)

            # Estimate time remaining
            tokens_remaining = tokens_to_train - tokens_trained
            eta_seconds = 0
            if tokens_remaining > 0:
                eta_seconds = tokens_remaining / tokens_per_second if tokens_per_second > 0 else 0
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))

            # Print formatted row
            print(f"{tokens_trained:<15} {tokens_per_second:<12.2f} {flops_per_second:<12.2e} {avg_loss:<10.4f} {perplexity:<12.2f} {eta_str:<10}")

            next_eval_tokens += eval_interval_tokens

        # Checkpointing
        if tokens_trained >= next_checkpoint:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tokens_trained': tokens_trained,
                'training_config': training_config,
            }, checkpoint_path)
            next_checkpoint += checkpoint_interval_tokens

    writer.close()
    print("\nTraining complete.\n")
    return model

def compute_perplexity(model, data_loader, eval_tokens=100000):
    """ 
    Compute the perplexity on the data in the data_loader over a specified number of tokens.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in data_loader:
            xb = batch[:, :-1].to(device)
            yb = batch[:, 1:].to(device)
            logits, loss = model(xb, yb)
            
            num_tokens = yb.numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            if total_tokens >= eval_tokens:
                break

    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss)

    model.train()
    return perplexity
