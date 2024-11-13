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
        xb, yb = next(train_loader)
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        if i % eval_interval == 0 and i > 0:
            perplexity = compute_perplexity(model, train_loader, eval_iters)
            perplexities.append(perplexity)
            print(f"Iteration {i}: Perplexity: {perplexity:.2f}")
        i += 1
    
    return model

def compute_perplexity(model, data_loader, eval_iters=100):
    """ 
    Compute the perplexity on the data in the data_loader.
    """
    model.eval()
    losses = []
    total_loss = 0
    for xb, yb in data_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred, loss = model(xb, yb)
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()

    model.train()
    return perplexity
