import torch

if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
    device = "mps"  # Apple Metal
else:
    device = "cpu"

print(f"Using device: {device}")