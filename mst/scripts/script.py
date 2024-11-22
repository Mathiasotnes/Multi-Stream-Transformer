##############################################################################
## script.py                                                                ##
## ---------                                                                ##
## Utilities for running scripts.                                           ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

import sys
import os
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

def load_config(config_file: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_file (str): Name of the configuration file (e.g., 'transformer.yaml').

    Returns:
        dict: Configuration dictionary.
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', config_file)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def print_training_details(checkpoint: dict) -> None:
    """
    Prints training details from the checkpoint in a nicely formatted style.
    If any information is missing, it prints 'Unknown' instead.
    """
    # Start building the string
    info_str = "Training Details:\n"

    # Extract training details from the checkpoint
    tokens_trained = checkpoint.get('tokens_trained', 'Unknown')
    total_flops = checkpoint.get('total_flops', 'Unknown')
    train_perplexity = checkpoint.get('train_perplexity', 'Unknown')
    val_perplexity = checkpoint.get('val_perplexity', 'Unknown')
    test_perplexity = checkpoint.get('test_perplexity', 'Unknown')

    # Format and append the details
    info_str += f" * Tokens Trained:        {tokens_trained}\n"
    info_str += f" * Total FLOPs:           {total_flops:.2e}\n"        if isinstance(total_flops, (int, float))        else f" * Total FLOPs:           {total_flops}\n"
    info_str += f" * Training PPL:          {train_perplexity:.3f}\n"   if isinstance(train_perplexity, (int, float))   else f" * Training PPL:          {train_perplexity}\n"
    info_str += f" * Validation PPL:        {val_perplexity:.3f}\n"     if isinstance(val_perplexity, (int, float))     else f" * Validation PPL:        {val_perplexity}\n"
    info_str += f" * Test PPL:              {test_perplexity:.3f}\n"    if isinstance(test_perplexity, (int, float))    else f" * Test PPL:              {test_perplexity}\n"
    info_str += f"{'='*40}\n"

    # Print the information
    print(info_str)

