##############################################################################
## train.py                                                                 ##
## --------                                                                 ##
## Training setup for language models.                                      ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

from typing import Any
from torch.utils.data import DataLoader
from itertools import cycle
import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model( model: Any, dataloader: DataLoader, training_config: dict ) -> Any:
    """Train a language model."""
    lr              = training_config["learning_rate"]
    eval_interval   = training_config["eval_interval"]
    eval_iters      = training_config["eval_iters"]
    tokens_to_train = training_config["tokens"]
    optimizer       = optim.Adam(model.parameters(), lr=lr)
    train_loader    = cycle(dataloader)
    tokens_trained  = 0
    i               = 0
    perplexities    = []
    
    while tokens_trained < tokens_to_train:
        # xb, yb = next(train_loader)
        batch = next(train_loader)
        xb = batch[:, :-1].to(device)
        yb = batch[:, 1:].to(device)
        
        optimizer.zero_grad()
        logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        
        tokens_trained += xb.numel()
        print(f"Tokens trained: {tokens_trained}/{tokens_to_train}", end="\r")
        
        if i % eval_interval == 0 and i > 0:
            perplexity = compute_perplexity(model, train_loader, eval_iters)
            perplexities.append(perplexity)
            print(f"\nIteration {i}: Perplexity: {perplexity:.2f}")
        i += 1
    
    print("\n\r")
    return model

def compute_perplexity(model, data_loader, eval_iters=100):
    """ 
    Compute the perplexity on the data in the data_loader.
    """
    model.eval()
    losses = []
    total_loss = 0
    for batch in data_loader:
        xb = batch[:, :-1].to(device)
        yb = batch[:, 1:].to(device)
        logits, loss = model(xb, yb)
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()

    model.train()
    return perplexity
