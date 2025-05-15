import torch
import transformers

from typing import Any, Dict, List, Optional, Sequence

from .. import conversation as conversation_lib
from ..constants import IGNORE_INDEX, SENTINEL_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_VI_START_TOKEN
from ..mm_utils import tokenizer_image_token
from ..utils.logging import logger

__all__ = [
    "tokenize_conversation",
]

DUMMY_CONVERSATION = [
    {"from": "human", "value": "question"},
    {"from": "gpt", "value": "answer"},
] * 10

def tokenize_conversation(
    messages: Sequence[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    add_generation_prompt: bool = False,
    overrides: Optional[Dict[str, str]] = None,
    no_system_prompt: bool = False,
    image_generation: bool = False,
    video_generation: bool = False,
) -> torch.Tensor:
    for message in messages:
        message["value"] = message["value"].strip()

    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    if no_system_prompt:
        conv.system = ""

    # Skip the first message if it is not from human
    if messages[0]["from"] != "human":
        messages = messages[1:]

    # Add a generation prompt if needed
    if add_generation_prompt:
        messages.append({"from": "gpt", "value": None})

    conv.messages = []
    for turn, message in enumerate(messages):
        role = roles[message["from"]]
        assert role == conv.roles[turn % 2]
        if overrides is not None and message["from"] in overrides:
            conv.append_message(role, overrides[message["from"]])
        else:
            conv.append_message(role, message["value"])
    
    prompt = conv.get_prompt()
    if image_generation:
        prompt += f" {DEFAULT_IM_START_TOKEN}"
    elif video_generation:
        prompt += f" {DEFAULT_VI_START_TOKEN}"
    else:
        pass

    return tokenizer_image_token(prompt, tokenizer, return_tensors="pt")

def _maybe_add_sentinel_token(tokenizer: transformers.PreTrainedTokenizer) -> None:
    if not hasattr(tokenizer, "sentinel_token"):
        tokenizer.add_tokens([SENTINEL_TOKEN], special_tokens=True)
        tokenizer.sentinel_token = SENTINEL_TOKEN
        tokenizer.sentinel_token_id = tokenizer.convert_tokens_to_ids(SENTINEL_TOKEN)

def infer_stop_tokens(tokenizer: transformers.PreTrainedTokenizer) -> List[str]:
    _maybe_add_sentinel_token(tokenizer)
    template = tokenize_conversation(DUMMY_CONVERSATION, tokenizer, overrides={"gpt": SENTINEL_TOKEN})

    stop_tokens = {tokenizer.eos_token}
    for k in range(template.size(0) - 1):
        if template[k] == tokenizer.sentinel_token_id:
            stop_token = tokenizer.decode(template[k + 1])
            stop_tokens.add(stop_token)
    return list(stop_tokens)
