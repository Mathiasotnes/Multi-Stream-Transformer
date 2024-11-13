##############################################################################
## __init__.py                                                              ##
## ------------                                                             ##
## Module initializer for the data package.                                 ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

from .data_loader import get_dataloader

__all__ = [
    "get_dataloader"
]