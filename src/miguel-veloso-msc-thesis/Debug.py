import torch

if torch.cuda.is_available():
    print("CUDA is available")
    print("CUDA Version:", torch.version.cuda)
    print("PyTorch Version:", torch.__version__)
else:
    print("CUDA is not available")
