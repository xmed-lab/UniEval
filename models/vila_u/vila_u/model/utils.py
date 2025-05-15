import os
import os.path as osp

from huggingface_hub import snapshot_download, repo_exists
from huggingface_hub.utils import HFValidationError
from transformers import  PretrainedConfig


def get_model_config(config):
    default_keys = ["llm_cfg", "vision_tower_cfg", "mm_projector_cfg"]
    
    if hasattr(config, "_name_or_path") and len(config._name_or_path) >= 2:
        root_path = config._name_or_path
    else:
        root_path = config.resume_path 

    if root_path is not None and not osp.exists(root_path):
        try:
            valid_hf_repo = repo_exists(root_path)
        except HFValidationError as e:
            valid_hf_repo = False
        if valid_hf_repo:
            root_path = snapshot_download(root_path)

    return_list = []
    for key in default_keys:
        cfg = getattr(config, key, None)
        if isinstance(cfg, dict):
            try:
                return_list.append(os.path.join(root_path, key[:-4]))
            except:
                raise ValueError(f"Cannot find resume path in config for {key}!")
        elif isinstance(cfg, PretrainedConfig):
            return_list.append(os.path.join(root_path, key[:-4]))
        elif isinstance(cfg, str):
            return_list.append(cfg)
        
    return return_list