##############################################################################
## transformer_training.py                                                  ##
## -----------------------                                                  ##
## Main program for training benchmark transformer model                    ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

from . import script
import torch
import os
import time
from mst import Transformer, get_dataloaders, train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def main() -> None:
    
    # Extract configurations
    config = script.load_config('transformer.yaml')
    model_config = config['model']
    dataset_config = config['dataset']
    training_config = config['training']

    # Set up data loaders
    train_dataloader, val_dataloader = get_dataloaders(
        dataset_name=dataset_config['name'],
        tokenizer_name=dataset_config['tokenizer_name'],
        max_seq_length=dataset_config['max_seq_length'],
        batch_size=dataset_config['batch_size'],
    )
    
    assert train_dataloader is not None, "Training dataloader not found."
    assert len(train_dataloader.dataset.tokenizer) == len(val_dataloader.dataset.tokenizer), "Tokenizer mismatch."
    
    # Create a transformer model
    model = Transformer(
        vocab_size=len(train_dataloader.dataset.tokenizer),
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        hidden_dim=model_config['d_ff'],
        num_blocks=model_config['num_layers'],
        dropout=model_config['dropout_rate'],
        echo_specs=True
    ).to(device)
    
    train_model(
        model=model,
        training_config=training_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    

if __name__ == "__main__":
    t = time.time()
    main()
    print(f"Program executed in: {time.time()-t:.2f} seconds.")
    exit(0)
