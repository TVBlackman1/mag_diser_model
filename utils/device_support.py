import torch

CPU_ONLY = True
def get_device():
    if CPU_ONLY:
        return "cpu"
    
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = "mps"  # Apple Metal
    else:
        device = "cpu"
    
    return device