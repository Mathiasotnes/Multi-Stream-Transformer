##############################################################################
## transformer.py                                                           ##
## --------------                                                           ##
## Main program for training and evaluating benchmark transformer model     ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

from . import script
import time

# Now that sys.path is adjusted, you can import from mst
from mst import Transformer, get_dataloader, train_model

def main() -> None:
    # Get configuration
    config = script.load_config('transformer.yaml')
    
    # Extract configurations
    model_config = config['model']
    dataset_config = config['dataset']
    training_config = config['training']

    # Set up data loaders
    dataloader = get_dataloader(
        dataset_name=dataset_config['name'],
        tokenizer_name=dataset_config['tokenizer_name'],
        max_seq_length=dataset_config['max_seq_length'],
        batch_size=dataset_config['batch_size'],
    )
    
    # Create a transformer model
    model = Transformer(
        vocab_size=len(dataloader.dataset.tokenizer),
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        hidden_dim=model_config['d_ff'],
        num_blocks=model_config['num_layers'],
        dropout=model_config['dropout_rate'],
        echo_specs=True
    )
    
    train_model(model, dataloader, training_config)
    

if __name__ == "__main__":
    t = time.time()
    main()
    print(f"Program executed in: {time.time()-t:.2f} seconds.")
    exit(0)
