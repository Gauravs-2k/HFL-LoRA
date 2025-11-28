import sys
print("Starting imports...", flush=True)
import torch
print(f"Torch version: {torch.__version__}", flush=True)
print("Checking CUDA...", flush=True)
if torch.cuda.is_available():
    print(f"CUDA is available. Device count: {torch.cuda.device_count()}", flush=True)
    print(f"Current device: {torch.cuda.current_device()}", flush=True)
    print(f"Device name: {torch.cuda.get_device_name(0)}", flush=True)
    try:
        x = torch.tensor([1.0]).cuda()
        print("Tensor allocation on GPU successful.", flush=True)
    except Exception as e:
        print(f"Tensor allocation failed: {e}", flush=True)
else:
    print("CUDA is NOT available.", flush=True)
print("Done.", flush=True)
