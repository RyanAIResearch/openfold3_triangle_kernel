import importlib.util

import torch


def is_turbopairformer_available() -> bool:
    """Check if turbopairformer is installed and CUDA is available."""
    return (
        importlib.util.find_spec("turbopairformer") is not None
        and torch.cuda.is_available()
    )
