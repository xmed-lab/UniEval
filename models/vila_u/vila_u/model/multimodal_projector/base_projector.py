import re
import torch.nn as nn
import torch

from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class MultimodalProjectorConfig(PretrainedConfig):
    model_type = "v2l_projector"

    def __init__(self, mm_projector_type: str=None, **kwargs):
        super().__init__()
        
        self.mm_projector_type = mm_projector_type


class MultimodalProjector(PreTrainedModel):
    config_class = MultimodalProjectorConfig

    def __init__(
        self, mm_projector_cfg: MultimodalProjectorConfig, config: PretrainedConfig
    ):
        super().__init__(mm_projector_cfg)
        mm_projector_type = mm_projector_cfg.mm_projector_type
        if mm_projector_type == "identity":
            self.layers = IdentityMap()
        elif mm_projector_type == "linear":
            self.layers = nn.Linear(config.mm_hidden_size, config.hidden_size)
        else:
            mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", mm_projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(config.hidden_size, config.hidden_size))
                self.layers = nn.Sequential(*modules)
            else:
                raise ValueError(f"Unknown projector type: {mm_projector_type}")

    def forward(self, x, *args, **kwargs):
        return self.layers(x)


AutoConfig.register("v2l_projector", MultimodalProjectorConfig)
AutoModel.register(MultimodalProjectorConfig, MultimodalProjector)