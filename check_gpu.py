# check_gpu.py
import torch

if torch.backends.mps.is_available():
    print("MPS backend is available!")
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print("Test tensor on MPS device:", x)
else:
    print("MPS backend is not available.")