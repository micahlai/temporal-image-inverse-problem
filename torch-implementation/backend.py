"""Explicit device/precision selection for the reconstruction backend."""

import torch


def resolve_device(device="auto"):
    if str(device) == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be auto, cpu, cuda[:index], or mps")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; install a CUDA-enabled PyTorch on an NVIDIA GPU host")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use an Apple Silicon Mac with an MPS-enabled PyTorch")
    return device


def resolve_dtype(dtype, device):
    if dtype is None or dtype == "auto":
        dtype = torch.float64 if device.type == "cpu" else torch.float32
    elif isinstance(dtype, str):
        dtype = {"float32": torch.float32, "float64": torch.float64}.get(dtype)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be auto, float32, or float64")
    if device.type == "mps" and dtype == torch.float64:
        raise ValueError("MPS does not support float64; use float32 or choose cpu/cuda")
    return dtype


def as_video(x):
    """Preserve tensor placement/precision; NumPy inputs become CPU float64."""
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.dtype not in (torch.float32, torch.float64):
            raise ValueError("Video tensors must use float32 or float64")
    else:
        x = torch.as_tensor(x, dtype=torch.float64)
    if x.ndim != 4 or not all(x.shape) or not torch.isfinite(x).all().item():
        raise ValueError("Expected a finite, nonempty (H, W, C, T) video")
    return x
