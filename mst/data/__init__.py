##############################################################################
## __init__.py                                                              ##
## ------------                                                             ##
## Module initializer for the data package.                                 ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

from .data_loader import get_dataloaders, get_tokenizer
from .open_web_text import get_owt_dataloaders
from .penn_treebank import get_ptb_dataloaders

__all__ = [
    "get_owt_dataloaders",
    "get_ptb_dataloaders",
    "get_dataloaders",
    "get_tokenizer"
]
