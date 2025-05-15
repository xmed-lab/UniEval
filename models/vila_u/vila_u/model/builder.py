import torch
from transformers import AutoConfig

from ..model import VILAULlamaModel
from ..constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_VI_START_TOKEN,
    DEFAULT_VI_END_TOKEN,
)


def load_pretrained_model(
    model_path,
    model_dtype=torch.bfloat16,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    config = AutoConfig.from_pretrained(model_path)
    config.resume_path = model_path
    config.model_dtype = model_dtype.__str__()

    model = VILAULlamaModel(
        config=config,
        low_cpu_mem_usage=True,
        **kwargs
    )
    tokenizer = model.tokenizer

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_vi_start_end = getattr(model.config, "mm_use_vi_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end and mm_use_vi_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN], special_tokens=True
        )
    elif mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    vision_tower = model.get_vision_tower()
    vision_tower.to(device=device, dtype=model_dtype)

    mm_projector = model.get_mm_projector()
    mm_projector.to(device=device, dtype=model_dtype)

    image_processor = vision_tower.image_processor

    if hasattr(model.llm.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len