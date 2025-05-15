import torch
import torch.nn as nn

from transformers import CLIPImageProcessor, PreTrainedModel, PretrainedConfig
from transformers.image_processing_utils import BaseImageProcessor

from .rqvaesigliptransformer import RQVAESIGLIPTransformerConfig, RQVAESIGLIPTransformer


class RQVAESIGLIPTransformerVisionTower(nn.Module):
    def __init__(self, model_name_or_path, config: PretrainedConfig):
        super().__init__()
        self.config = RQVAESIGLIPTransformerConfig.from_pretrained(model_name_or_path)
        self.vision_tower = RQVAESIGLIPTransformer.from_pretrained(model_name_or_path, torch_dtype=eval(config.model_dtype))
        self.is_loaded = True

        if self.config.hidden_size == 1152:
            self.image_processor = CLIPImageProcessor(
                size={"height": 384, "width": 384}, 
                crop_size={"height": 384, "width": 384}, 
                image_mean=[0.5, 0.5, 0.5], 
                image_std=[0.5, 0.5, 0.5]
            )
            self.image_tokens = 729
        elif self.config.hidden_size == 1024:
            self.image_processor = CLIPImageProcessor(
                size={"height": 256, "width": 256}, 
                crop_size={"height": 256, "width": 256}, 
                image_mean=[0.5, 0.5, 0.5], 
                image_std=[0.5, 0.5, 0.5]
            )
            self.image_tokens = 256
        else:
            raise NotImplementedError()
    
    def forward(self, images, text_vocab_size):
        output = self.vision_tower.rqvaesiglip.encode_image(images)
        image_features, tokens = output[-1], output[-2]

        bs, patch_size, _, dim = image_features.shape
        image_features = torch.reshape(image_features, [bs, patch_size**2, dim])
        tokens = torch.add(torch.reshape(tokens, [bs, patch_size**2, -1]), text_vocab_size)

        return image_features, tokens