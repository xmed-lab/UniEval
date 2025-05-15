from transformers import PretrainedConfig


class VILAUConfig(PretrainedConfig):
    model_type = "vila_u"

    def __init__(
        self,
        llm_cfg=None,
        vision_tower_cfg=None,
        mm_projector_cfg=None,
        architectures=None,
        resume_path=None,
        hidden_size=None,
        mm_hidden_size=None,
        image_aspect_ratio=None,
        num_video_frames=None,
        mm_use_im_start_end=False,
        mm_use_vi_start_end=False,
        mm_use_im_patch_token=True,
        **kwargs
    ):
        super().__init__()

        self.llm_cfg = llm_cfg
        self.vision_tower_cfg = vision_tower_cfg
        self.mm_projector_cfg = mm_projector_cfg
        self.architectures = architectures
        self.resume_path = resume_path
        self.hidden_size = hidden_size
        self.mm_hidden_size = mm_hidden_size
        self.image_aspect_ratio = image_aspect_ratio
        self.num_video_frames = num_video_frames
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_vi_start_end = mm_use_vi_start_end
        self.mm_use_im_patch_token = mm_use_im_patch_token