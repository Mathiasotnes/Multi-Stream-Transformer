##############################################################################
## __init__.py                                                              ##
## ------------                                                             ##
## Module initializer for the Multi-Stream Transformer package.             ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

from .architecture import Transformer
from .data import get_dataloader, get_tokenizer
from .training import train_model, compute_perplexity

__all__ = [
    "Transformer",
    "get_dataloader",
    "get_tokenizer",
    "train_model",
    "compute_perplexity"
]

