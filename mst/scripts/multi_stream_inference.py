##############################################################################
## multi_stream_inference.py                                                ##
## -------------------------                                                ##
## Main program for loading and generating text with a multi-stream         ##
## transformer model from a checkpoint with token tracking                  ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

from colorama import Fore
from . import script
import torch
from mst import MultiStreamTransformer, get_tokenizer
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main() -> None:
    # Load configurations
    config = script.load_config('dual_stream.yaml') # Use dual_stream.yaml, triple_stream.yaml or multi_stream.yaml
    model_config = config['model']
    dataset_config = config['dataset']
    training_config = config['training']

    checkpoint_dir = training_config['checkpoint_dir']
    model_name = training_config['model_name']

    # Set checkpoint path
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pt")

    # Initialize transformer model
    tokenizer = get_tokenizer(dataset_config['tokenizer_name'])
    model = MultiStreamTransformer(
        vocab_size=len(tokenizer),
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        hidden_dim=model_config['d_ff'],
        num_streams=model_config['num_streams'],
        blocks_per_stream=model_config['blocks_per_stream'],
        num_blocks=model_config['num_blocks'],
        dropout=model_config['dropout_rate'],
        fusion_method=model_config['fusion_method'],
        echo_specs=True
    ).to(device)

    # Load model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    script.print_training_details(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Extract tokens trained
    tokens_trained = checkpoint.get('tokens_trained', None)
    if tokens_trained is not None:
        print(f"Model has been trained on {tokens_trained} tokens.")
    else:
        print("Token count information not found in the checkpoint.")

    model.eval()
    print("Model is ready. Type a prompt and press enter to receive a response.")

    # Infinite prompt loop
    while True:
        prompt = input(f"\r\n{Fore.WHITE}> ")

        if prompt.lower() in {"exit", "quit"}:
            print("Exiting the program.")
            break
        
        elif prompt == "":
            continue

        # Tokenize the input prompt and generate response
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=50,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.1
            )

        response_ids = output_ids[0, input_ids.size(1):]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        print(f"Model: {Fore.CYAN} {response}")

if __name__ == "__main__":
    t = time.time()
    main()
    print(f"Program executed in: {time.time()-t:.2f} seconds.")
    exit(0)
