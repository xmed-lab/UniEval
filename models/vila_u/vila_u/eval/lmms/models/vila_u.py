import accelerate
import os
import requests
import torch

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from tqdm import tqdm
from typing import List, Tuple

import vila_u
from vila_u import conversation as conversation_lib
from vila_u.media import Video
from vila_u.utils import distributed as dist
from vila_u.utils import io


@register_model("vila_u")
class VILA_U(lmms):
    def __init__(
        self, model_path: str, conv_mode: str, num_video_frames: int = 8, batch_size: int = 1
    ) -> None:
        super().__init__()
        assert batch_size == 1, "VILA-U only supports batch size of 1 at the moment."
        self._update_gpt_eval_model()

        devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
        torch.cuda.set_device(devices[0])

        self.model = vila_u.load(model_path, devices=devices)
        self.model.config.num_video_frames = num_video_frames
        context_length = num_video_frames * 512
        self.model.config.model_max_length = context_length
        self.model.config.tokenizer_model_max_length = context_length
        self.model.llm.config.model_max_length = context_length
        self.model.llm.config.tokenizer_model_max_length = context_length
        self.model.tokenizer.model_max_length = context_length

        conversation_lib.default_conversation = conversation_lib.conv_templates[conv_mode].copy()

        self.accelerator = accelerate.Accelerator()
        self.device = torch.device(f"cuda:{devices[0]}")
        self._world_size = dist.size()
        self._rank = dist.rank()

    def _update_gpt_eval_model(self) -> None:
        _unpatched_post = requests.post

        def _patched_post(url, json, **kwargs):
            if json is not None and "model" in json:
                if json["model"] == "gpt-3.5-turbo-0613":
                    json["model"] = "gpt-4o-mini"
            return _unpatched_post(url, json=json, **kwargs)

        requests.post = _patched_post

    def generate_until(self, requests: List[Instance]) -> List[str]:
        responses = []
        for request in tqdm(requests, disable=not dist.is_main()):
            prompt, generation_kwargs, doc_to_visual, doc_id, task, split = self._patch(request.args)
            doc = self.task_dict[task][split][doc_id]

            # Generate multimodal prompt
            medias = []
            for media in doc_to_visual(doc):
                if isinstance(media, str):
                    if any(media.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]):
                        media = Video(media)
                    else:
                        raise NotImplementedError(f"Unsupported media type: {media}")
                medias.append(media)
            prompt = medias + [prompt]

            # Override generation config
            generation_config = self.model.default_generation_config
            generation_config.update(**generation_kwargs)

            # Generate and cache response
            cache_path = None
            if "CACHE_DIR" in os.environ:
                cache_path = os.path.join(os.environ["CACHE_DIR"], f"{task}_{split}_{doc_id}.txt")

            if cache_path is not None and os.path.exists(cache_path):
                response = io.load(cache_path)
            else:
                response = self.model.generate_content(prompt, generation_config=generation_config)
                if cache_path is not None:
                    io.save(cache_path, response)
            responses.append(response)

            print("Prompt:", prompt)
            print("Response:", response)
        return responses

    def _patch(self, args: Tuple) -> Tuple:
        prompt, generation_kwargs, doc_to_visual, doc_id, task, split = args
        doc = self.task_dict[task][split][doc_id]

        return prompt, generation_kwargs, doc_to_visual, doc_id, task, split

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError
