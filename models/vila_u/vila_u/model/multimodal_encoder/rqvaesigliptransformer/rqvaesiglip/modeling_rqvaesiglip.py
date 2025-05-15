import torch

from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, AutoConfig, AutoModel
from typing import Optional

from .configuration_rqvaesiglip import RQVAESiglipConfig
from .modules import Decoder
from .quantizations import RQBottleneck
from .siglip import SiglipModel


class RQVAESiglipModel(PreTrainedModel):
    config_class = RQVAESiglipConfig
    def __init__(self, config: RQVAESiglipConfig):
        super().__init__(config)

        siglip_config = SiglipModel.config_class.from_pretrained(config.pretrained_model)
        self.siglip_model = SiglipModel._from_config(siglip_config)

        self.quantizer = RQBottleneck(
            latent_shape=config.latent_shape,
            code_shape=config.code_shape,
            n_embed=config.n_embed,
            decay=config.decay,
            shared_codebook=config.shared_codebook,
            restart_unused_codes=config.restart_unused_codes,
        )
        self.post_quant_conv = torch.nn.Conv2d(config.embed_dim, config.ddconfig["z_channels"], 1)

        self.decoder = Decoder(**config.ddconfig)

        try:
            self.decoder_latent_shape = config.decoder_latent_shape
        except:
            self.decoder_latent_shape = None
        
        self.logit_scale = self.siglip_model.logit_scale
        self.logit_bias = self.siglip_model.logit_bias
        
    def encode_image(self, image):
        vision_model = self.siglip_model.vision_model
        hidden_states = vision_model.embeddings(image)

        attention_mask = None
        output_attentions = None
        for i, encoder_layer in enumerate(vision_model.encoder.layers):
            if vision_model.encoder.gradient_checkpointing and vision_model.encoder.training:
                layer_outputs = vision_model.encoder._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )
            hidden_states = layer_outputs[0]
            if i == len(vision_model.encoder.layers) - 2:
                B, L, C = hidden_states.shape
                hidden_states = hidden_states.reshape(B, int(L**0.5), int(L**0.5), C)
                z_q, quant_loss, code = self.quantizer(hidden_states)
                
                return code, z_q

    def decode(self, z_q):
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        if self.decoder_latent_shape is not None:
            z_q = F.interpolate(z_q.to(torch.float32), size=tuple(self.decoder_latent_shape), mode='bilinear').to(torch.bfloat16)
            
        z_q = self.post_quant_conv(z_q)
        out = self.decoder(z_q)

        return out
    
    @torch.no_grad()
    def get_code_emb_with_depth(self, code):
        return self.quantizer.embed_code_with_depth(code)


AutoConfig.register("rqvaesiglip_model", RQVAESiglipConfig)
AutoModel.register(RQVAESiglipConfig, RQVAESiglipModel)