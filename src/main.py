import torch
import numpy as np

def main():
    print("*** Self-Tuning networks ***")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")

if __name__ == "__main__":
    main()