from typing import Any, List

__all__ = ["make_list", "disable_torch_init"]


def make_list(obj: Any) -> List:
    return obj if isinstance(obj, list) else [obj]


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)