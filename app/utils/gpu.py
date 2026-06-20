"""
GPU utility helpers.

Used to keep cheap CPU deploy tests clean while allowing
the same container to run full inference on a GPU server.
"""

import torch


def gpu_is_available() -> bool:
    """
    Returns True when CUDA is available to PyTorch.
    """
    return torch.cuda.is_available()


def gpu_device_name() -> str | None:
    """
    Returns the active CUDA device name when available.
    """
    if not gpu_is_available():
        return None

    return torch.cuda.get_device_name(0)