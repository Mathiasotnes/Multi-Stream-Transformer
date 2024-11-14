##############################################################################
## transformer_test.py                                                      ##
## -------------------                                                      ##
## Main program for testing transformer perplexity on test set              ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

from colorama import Fore
from . import script
import torch
from mst import Transformer, get_tokenizer, get_dataloader, compute_perplexity
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main() -> None:
    # Load configurations
    config = script.load_config('transformer.yaml')
    model_config = config['model']
    dataset_config = config['dataset']
    training_config = config['training']

    checkpoint_dir = training_config['checkpoint_dir']
    model_name = training_config['model_name']

    # Set checkpoint path
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{model_name}.pt")

    # Initialize transformer model
    tokenizer = get_tokenizer(dataset_config['tokenizer_name'])
    model = Transformer(
        vocab_size=len(tokenizer),
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        hidden_dim=model_config['d_ff'],
        num_blocks=model_config['num_layers'],
        dropout=model_config['dropout_rate'],
        echo_specs=True
    ).to(device)

    # Load model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Extract tokens trained
    tokens_trained = checkpoint.get('tokens_trained', None)
    if tokens_trained is not None:
        print(f"Model has been trained on {tokens_trained} tokens.")
    else:
        print("Token count information not found in the checkpoint.")

    model.eval()
    
    # Set up test dataloader
    test_dataloader = get_dataloader(
        dataset_name=dataset_config['name'],
        tokenizer_name=dataset_config['tokenizer_name'],
        max_seq_length=dataset_config['max_seq_length'],
        batch_size=dataset_config['batch_size'],
        split='test'
    )
    
    # Count number of tokens in test set
    test_tokens = 0
    for batch in test_dataloader:
        test_tokens += batch.numel()
        
    print(f"Number of tokens in test set: {test_tokens}")
    
    # Compute perplexity
    perplexity = compute_perplexity( model, test_dataloader, test_tokens )
    print(f"{'='*40}")
    print(f"{'Test Perplexity:':<20} {Fore.CYAN} {perplexity:.2f} {Fore.WHITE}")
    print(f"{'='*40}")
    
if __name__ == "__main__":
    t = time.time()
    main()
    print(f"Program executed in: {time.time()-t:.2f} seconds.")
    exit(0)
