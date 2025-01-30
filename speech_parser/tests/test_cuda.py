import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Supports float16: {torch.cuda.is_bf16_supported()}")
    print(f"Supports float32: {True}")  # All GPUs support float32
else:
    print("CUDA is not available.")
