##############################################################################
## main.py                                                                  ##
## -------                                                                  ##
## Main program for prototyping and testing the code.                       ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

from mst.models import transformer
import time

def main() -> None:
    
    # Create a transformer model
    model = transformer.Transformer(
        num_heads=8,
        d_model=512,
        hidden_dim=2048,
        num_layers=6,
        num_classes=10,
        seq_length=128,
        vocab_size=100,
        max_length=512,
        dropout=0.1
    )
    print(model)

if __name__ == "__main__":
    t = time.time()
    main()
    print(f"Program executed in: {time.time()-t:.2f} seconds.")
    exit(0)