import torch, os
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (Torch):", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")