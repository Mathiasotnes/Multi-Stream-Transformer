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
from .data import get_dataloaders, get_tokenizer
from .training import train_model

__all__ = [
    "Transformer",
    "get_dataloaders",
    "get_tokenizer",
    "train_model"
]

