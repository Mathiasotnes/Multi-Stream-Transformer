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
