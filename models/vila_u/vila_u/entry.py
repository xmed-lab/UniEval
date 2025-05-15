import os
import torch
import typing
import sys
from typing import List, Optional

if typing.TYPE_CHECKING:
    from transformers import PreTrainedModel
else:
    PreTrainedModel = None

from .conversation import auto_set_conversation_mode
from .model.builder import load_pretrained_model

__all__ = ["load"]


def load(
    model_path: str,
    devices: Optional[List[int]] = None,
    **kwargs,
) -> PreTrainedModel:
    auto_set_conversation_mode(model_path)

    model_path = os.path.expanduser(model_path)
    if os.path.exists(os.path.join(model_path, "model")):
        model_path = os.path.join(model_path, "model")

    # Set `max_memory` to constrain which GPUs to use
    if devices is not None:
        assert "max_memory" not in kwargs, "`max_memory` should not be set when `devices` is set"
        kwargs.update(max_memory={device: torch.cuda.get_device_properties(device).total_memory for device in devices})

    model = load_pretrained_model(model_path, **kwargs)[1]

    return model