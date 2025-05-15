import transformers

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class DataArguments:
    data_mixture: str = "llava_1_5_mm_align"
    image_aspect_ratio: str = "square"
    lazy_preprocess: bool = False
    vflan_no_system_prompt: bool = False
    num_video_frames: int = 8


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    mm_projector: Optional[str] = field(default="mlp2x_gelu")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_vi_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    interpolate_mode: Optional[str] = field(default="linear")
    drop_path_rate: Optional[float] = field(default=0.)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    tune_vision_tower: bool = field(default=False)
    tune_language_model: bool = field(default=False)
    tune_mm_projector: bool = field(default=False)
    chunk_sampler: bool = field(default=False)
    model_dtype: str = field(default="torch.bfloat16")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    total_time_limit: int = field(
        default=-1, metadata={"help": "Timeout limit for this job (in minutes)."}
    )
    pre_terminate_time: int = field(
        default=10,
        metadata={
            "help": "Time to terminate the task inadvance (minutes), saveing checkpoints needs time."
        },
    )