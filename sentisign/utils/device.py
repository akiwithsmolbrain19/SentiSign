"""
Device utilities for PyTorch.

Handles automatic detection and selection of compute devices (CUDA, MPS, CPU).
"""

import torch


def get_device(device_preference: str = "auto") -> torch.device:
    """Get the appropriate torch device.

    Args:
        device_preference: One of "auto", "cuda", "mps", or "cpu"
            - "auto": Automatically select best available device
            - "cuda": Force CUDA (will fail if not available)
            - "mps": Force MPS (Apple Silicon, will fail if not available)
            - "cpu": Force CPU

    Returns:
        torch.device object

    Examples:
        >>> device = get_device("auto")  # Auto-detect best device
        >>> device = get_device("cpu")   # Force CPU
    """
    if device_preference == "auto":
        # Priority: CUDA > MPS > CPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_preference)


def print_device_info(device: torch.device) -> None:
    """Print information about the selected device.

    Args:
        device: torch.device object
    """
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif device.type == "mps":
        print("  Apple Silicon GPU (Metal Performance Shaders)")
    else:
        print("  CPU")
