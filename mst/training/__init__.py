##############################################################################
## __init__.py                                                              ##
## ------------                                                             ##
## Module initializer for the training package.                             ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

from .train import train_model, compute_perplexity

__all__ = [
    "train_model",
    "compute_perplexity"
]
