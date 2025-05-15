import os

from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from .rqvaesigliptransformer_encoder import RQVAESIGLIPTransformerVisionTower


def build_vision_tower(
    model_name_or_path: str, config: PretrainedConfig
) -> PreTrainedModel:
    if model_name_or_path is None:
        return None

    vision_tower_arch = None
    if config.resume_path:
        assert os.path.exists(
            model_name_or_path
        ), f"Resume vision tower path {model_name_or_path} does not exist!"
        vision_tower_cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        vision_tower_arch = vision_tower_cfg.architectures[0].lower()
    vision_tower_name = (
        vision_tower_arch if vision_tower_arch is not None else model_name_or_path
    )

    vision_tower = RQVAESIGLIPTransformerVisionTower(model_name_or_path, config)

    config.mm_hidden_size = vision_tower.config.hidden_size
    
    return vision_tower