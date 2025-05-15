"""
    Modified from https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/models/rqtransformer/transformers.py.
"""

import torch
import torch.nn as nn

from collections import OrderedDict
from torch.nn import functional as F
from transformers import PreTrainedModel, AutoConfig, AutoModel

from .attention import AttentionStack
from .configuration_rqtransformer import RQTransformerConfig, AttentionStackConfig, AttentionBlockConfig


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')

    return out


def top_p_probs(probs, p):    
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_idx_remove_cond = cum_probs >= p
    
    sorted_idx_remove_cond[..., 1:] = sorted_idx_remove_cond[..., :-1].clone()
    sorted_idx_remove_cond[..., 0] = 0
    
    indices_to_remove = sorted_idx_remove_cond.scatter(-1, sorted_indices, sorted_idx_remove_cond)
    probs = probs.masked_fill(indices_to_remove, 0.0)
    norm_probs = probs / torch.sum(probs, dim=-1, keepdim=True)

    return norm_probs


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None):
    """Take a 2-dim tensor, apply softmax along each row, and sample from
    each multinomial distribution defined by the rows.

    Args:
        logits: 2-dim tensor of shape (n_samples, logit_dim)
        temperature (float): softmax temperature
        top_k (Optional[int]): if given, sample only using `top_k` logits
        top_p (Optional[float]): if given, sample only using `top_p` logits

    Returns:
        samples: 1-dim integer tensor of shape (n_samples,)
    """

    logits = logits.to(dtype=torch.float32)
    logits = logits / temperature

    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    
    if torch.sum(torch.isnan(logits)):
        print('WARNING... NaN observed')
        logits[torch.isnan(logits)] = -float('Inf')

    probs = F.softmax(logits, dim=-1)
    
    if top_p is not None:
        probs = top_p_probs(probs, top_p)

    try:
        samples = torch.multinomial(probs, num_samples=1)
    except:
        raise RuntimeError

    return samples.view(-1)


class RQTransformer(PreTrainedModel):
    config_class = RQTransformerConfig
    def __init__(self, config: RQTransformerConfig):
        super().__init__(config)
        self.in_mlp_1 = nn.Linear(config.input_embed_dim_1, config.embed_dim)
        self.in_mlp_2 = nn.Linear(config.input_embed_dim_2, config.embed_dim)

        blockconfig = AttentionBlockConfig(embed_dim=config.embed_dim, n_head=config.head["block"]["n_head"])
        stackconfig = AttentionStackConfig(n_layer=config.head["n_layer"], block=blockconfig)
        self.head_transformer = AttentionStack(stackconfig)

        self.pos_emb_d = nn.Parameter(torch.zeros(1, config.block_size[2], config.embed_dim))
        self.pos_emb_d.data.normal_(mean=0.0, std=0.02)

        self.classifier_mlp = nn.Sequential(OrderedDict([
            ('layer_norm', nn.LayerNorm(config.embed_dim)),
            ('linear', nn.Linear(config.embed_dim, config.vocab_size)),
        ]))
    
    def embed_with_model_aux(self, code, model_aux):
        xs_emb, _ = model_aux.get_code_emb_with_depth(code)
        return xs_emb

    def forward(self, embed_from_body, code, model_aux=None):
        B, seq_len, D = code.shape

        depth_ctx = self.embed_with_model_aux(code, model_aux)
        depth_ctx = torch.cumsum(depth_ctx, dim=-2)
        depth_ctx = self.in_mlp_1(depth_ctx)

        embed_from_body = self.in_mlp_2(embed_from_body)

        depth_ctx_full = torch.cat(
            [
                embed_from_body.view(B, seq_len, 1, -1),
                depth_ctx[:, :, :-1, :],
            ],
            dim=-2,
        )

        depth_ctx_full = depth_ctx_full.reshape(B * seq_len, D, -1)
        depth_ctx_full = depth_ctx_full + self.pos_emb_d[:, :D, :]

        head_outputs = self.head_transformer(depth_ctx_full)
        head_outputs = head_outputs.reshape(B, seq_len, D, -1)
        head_outputs = self.classifier_mlp(head_outputs)

        return head_outputs
    
    def generate(self, embed_from_body, model_aux=None, cfg=3.0):
        generate_idx = 1
        B, seq_len, _ = embed_from_body.shape

        embed_from_body = self.in_mlp_2(embed_from_body)

        depth_ctx_full = embed_from_body.view(B, seq_len, 1, -1)
        depth_ctx_full = depth_ctx_full.reshape(B * seq_len, generate_idx, -1)
        depth_ctx_full = depth_ctx_full + self.pos_emb_d[:, :generate_idx, :]

        head_outputs = self.head_transformer(depth_ctx_full)
        head_outputs = head_outputs.reshape(B, -1)

        logits = self.classifier_mlp(head_outputs)

        logits = logits[B//2:, :] + cfg * (logits[:B//2, :] - logits[B//2:, :])
        code = sample_from_logits(logits, temperature=1.0, top_p=0.96, top_k=900)
        code = code.reshape(B//2, seq_len, 1).repeat(2, 1, self.pos_emb_d.shape[1])

        for i in range(self.pos_emb_d.shape[1]-1):
            generate_idx += 1
            depth_ctx = self.embed_with_model_aux(code, model_aux)
            depth_ctx = torch.cumsum(depth_ctx, dim=-2)[:, :, :i+1, :]
            if len(depth_ctx.shape) == 3:
                depth_ctx = depth_ctx.unsqueeze(2)
            depth_ctx = self.in_mlp_1(depth_ctx)

            depth_ctx_full = torch.cat(
                [
                    embed_from_body.view(B, seq_len, 1, -1),
                    depth_ctx,
                ],
                dim=-2,
            )

            depth_ctx_full = depth_ctx_full.reshape(B * seq_len, generate_idx, -1)
            depth_ctx_full = depth_ctx_full + self.pos_emb_d[:, :generate_idx, :]

            head_outputs = self.head_transformer(depth_ctx_full)
            head_outputs = head_outputs[:, -1, :]

            logits = self.classifier_mlp(head_outputs)

            logits = logits[B//2:, :] + cfg * (logits[:B//2, :] - logits[B//2:, :])
            code_generate = sample_from_logits(logits, temperature=1.0, top_p=0.96, top_k=900)
            code_generate = code_generate.reshape(B//2, seq_len).repeat(2, 1)
            code[:, :, i+1] = code_generate

        out_features = self.embed_with_model_aux(code, model_aux)
        out_features = torch.cumsum(out_features, dim=-2)[:, :, -1, :]

        return out_features, code


AutoConfig.register("rqtransformer_model", RQTransformerConfig)
AutoModel.register(RQTransformerConfig, RQTransformer)