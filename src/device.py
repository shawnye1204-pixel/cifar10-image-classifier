import torch

from . import config


def get_device():
    """Select an available device or validate an explicit request."""
    requested = config.DEVICE.lower()
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }

    if requested == "auto":
        # Prefer NVIDIA GPU, then Mac GPU, with CPU as the fallback.
        for name in ("cuda", "mps", "cpu"):
            if available[name]:
                return torch.device(name)

    if requested not in available:
        raise ValueError(f"Unknown device: {requested}")
    if not available[requested]:
        raise RuntimeError(f"Requested device '{requested}' is unavailable.")

    return torch.device(requested)
