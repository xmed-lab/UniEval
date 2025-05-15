import copy
import logging
import os
import os.path as osp
import torch
import warnings

from abc import ABC
from collections import OrderedDict
from transformers import AutoConfig, GenerationConfig, PreTrainedModel
from transformers.modeling_utils import ContextManagers, no_init_weights
from typing import List, Optional, Union

from ..constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_VI_START_TOKEN,
    DEFAULT_VI_END_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from ..model.configuration_vila_u import VILAUConfig
from ..model.language_model.builder import build_llm_and_tokenizer
from ..model.multimodal_encoder.builder import build_vision_tower
from ..model.multimodal_encoder.rqvaesigliptransformer_encoder import RQVAESIGLIPTransformerVisionTower
from ..model.multimodal_projector.builder import build_mm_projector
from ..model.utils import get_model_config
from ..mm_utils import process_images
from ..utils.media import extract_media
from ..utils.tokenizer import infer_stop_tokens, tokenize_conversation


class VILAUMetaModel(ABC):
    def init_vlm(self, config: PreTrainedModel = None, *args, **kwargs):
        if hasattr(self, "llm") or hasattr(self, "vision_tower")  or hasattr(self, "mm_projector"):
            return 
        
        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype

        cfgs = get_model_config(config)
        if len(cfgs) == 3:
            llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
        else:
            raise ValueError("`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config.")

        self.llm, self.tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        self.vision_tower = build_vision_tower(vision_tower_cfg, config)
        self.mm_projector = build_mm_projector(mm_projector_cfg, config)

        self.post_config()
        self.is_loaded = True

        assert (
            self.llm is not None or self.vision_tower is not None or self.mm_projector is not None
        ), "At least one of the components must be instantiated."

    @classmethod
    def load_pretrained(cls, model_path_or_config, *args, **kwargs):
        kwargs.pop("config", None)

        if isinstance(model_path_or_config, str):
            config = AutoConfig.from_pretrained(model_path_or_config)
        elif isinstance(model_path_or_config, VILAUConfig):
            config = model_path_or_config
        else:
            raise NotImplementedError(f"wrong type, {type(model_path_or_config)} \
                                      {isinstance(model_path_or_config, VILAUConfig)}")

        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype
        
        cfgs = get_model_config(config)
        if len(cfgs) == 3:
            llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
        else:
            raise ValueError("`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config.")

        with ContextManagers([no_init_weights(_enable=True),]):
            vlm = cls(config, *args, **kwargs)
        
        if hasattr(vlm, "llm") or hasattr(vlm, "vision_tower")  or hasattr(vlm, "mm_projector"):
            if vlm.is_loaded:
                return vlm
        
        vlm.llm, vlm.tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        vlm.vision_tower = build_vision_tower(vision_tower_cfg, config)
        vlm.mm_projector = build_mm_projector(mm_projector_cfg, config)

        self.post_config()
        self.is_loaded = True

        assert (
            vlm.llm is not None or vlm.vision_tower is not None or vlm.mm_projector is not None
        ), "At least one of the components must be instantiated."
        return vlm

    def save_pretrained(self, output_dir, state_dict=None):
        if state_dict is None:
            state_dict = self.state_dict()
        
        if getattr(self, "tokenizer", None):
            self.tokenizer.save_pretrained(osp.join(output_dir, "llm"))

        if self.get_llm():
            print(f"saving llm to {osp.join(output_dir, 'llm')}")
            self.llm.config._name_or_path = osp.join(output_dir, "llm")
            llm_state_dict = OrderedDict({k.split("llm.")[-1]: v for k, v in state_dict.items() if "llm" in k})
            self.llm.save_pretrained(os.path.join(output_dir, "llm"), state_dict=llm_state_dict)
            self.config.llm_cfg = self.llm.config

        if self.get_vision_tower() and "radio" not in self.get_vision_tower().__class__.__name__.lower():
            print(f"saving vision_tower to {osp.join(output_dir, 'vision_tower')}")
            self.vision_tower.config._name_or_path = osp.join(output_dir, "vision_tower")
            vision_tower_state_dict = OrderedDict(
                {k.split("vision_tower.vision_tower.")[-1]: v for k, v in state_dict.items() if "vision_tower" in k}
            )
            self.vision_tower.vision_tower.save_pretrained(
                os.path.join(output_dir, "vision_tower"),
                state_dict=vision_tower_state_dict,
            )
            self.vision_tower.image_processor.save_pretrained(os.path.join(output_dir, "vision_tower"))
            self.config.vision_tower_cfg = self.vision_tower.config
            if hasattr(self.config.vision_tower_cfg, 'auto_map'):
                delattr(self.config.vision_tower_cfg, 'auto_map')

        if self.get_mm_projector():
            print(f"saving mm_projector to {osp.join(output_dir, 'mm_projector')}")
            self.mm_projector.config._name_or_path = osp.join(output_dir, "mm_projector")
            mm_projector_state_dict = OrderedDict(
                {k.split("mm_projector.")[-1]: v for k, v in state_dict.items() if "mm_projector" in k}
            )
            self.mm_projector.save_pretrained(
                os.path.join(output_dir, "mm_projector"),
                state_dict=mm_projector_state_dict,
            )
            self.config.mm_projector_cfg = self.mm_projector.config

        self.config._name_or_path = output_dir
        self.config.architectures = [self.__class__.__name__]
        self.config.save_pretrained(output_dir)

    def get_llm(self):
        llm = getattr(self, "llm", None)
        if type(llm) is list:
            llm = llm[0]
        return llm

    def get_lm_head(self):
        lm_head = getattr(self.get_llm(), "lm_head", None)
        return lm_head

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_mm_projector(self):
        mm_projector = getattr(self, "mm_projector", None)
        if type(mm_projector) is list:
            mm_projector = mm_projector[0]
        return mm_projector

    def post_config(self):
        self.training = self.get_llm().training

        if getattr(self.config, "llm_cfg", None) is None:
            self.config.llm_cfg = self.llm.config
        if getattr(self.config, "vision_tower_cfg", None) is None:
            self.config.vision_tower_cfg = self.vision_tower.config
        if getattr(self.config, "mm_projector_cfg", None) is None:
            self.config.mm_projector_cfg = self.mm_projector.config

    def freezed_module_patch(self):
        '''
        Huggingface will call model.train() at each training_step. To ensure the expected behaviors for modules like dropout, batchnorm, etc., we need to call model.eval() for the freezed modules.
        '''
        if self.training:
            if self.get_llm() and not getattr(self.config, "tune_language_model", False):
                logging.warning("Caution: Your LLM is currently in training mode, ensuring accurate gradient computation. Please be vigilant, particularly regarding BatchNorm and Dropout operations.")
            if self.get_vision_tower() and not getattr(self.config, "tune_vision_tower", False):
                self.get_vision_tower().eval()
            if self.get_mm_projector() and not getattr(self.config, "tune_mm_projector", False):
                self.get_mm_projector().eval()
    
    def encode_images(self, images, image_ids):
        vision_tower = self.get_vision_tower()
        mm_projector = self.get_mm_projector()

        if isinstance(vision_tower, RQVAESIGLIPTransformerVisionTower):
            vision_tower.vision_tower.rqvaesiglip.eval()
        else:
            raise NotImplementedError()

        image_features, tokens = vision_tower(images, self.llm.vocab_size)
        image_features = mm_projector(image_features)

        return image_features, tokens
    
    def _temporary_reorder_cache(self, past_key_values, sorted_idx):
        return self.get_llm()._temporary_reorder_cache(past_key_values, sorted_idx)

    def get_input_embeddings(self):
        return self.get_llm().get_input_embeddings()

    def get_output_embeddings(self):
        return self.get_llm().get_output_embeddings()

    def resize_token_embeddings(self, embed_size):
        self.get_llm().resize_token_embeddings(embed_size)


class VILAUMetaForCausalLM(ABC):
    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (
                                attention_mask.shape[0],
                                target_shape - attention_mask.shape[1],
                            ),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        if type(images) is list:
            images = torch.cat(images, dim=0)
        elif images.ndim == 5:
            images = images.flatten(0, 1)

        input_image_ids = input_ids[input_ids == IMAGE_TOKEN_INDEX]
        image_features, tokens = self.encode_images(images, input_image_ids)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids_copy = input_ids.clone()
        input_ids_copy[input_ids_copy == IMAGE_TOKEN_INDEX] = 0
        input_embeds = self.llm.model.embed_tokens(input_ids_copy)

        input_ids = [
            cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        input_embeds_1 = [
            cur_input_embeds[cur_attention_mask]
            for cur_input_embeds, cur_attention_mask in zip(input_embeds, attention_mask)
        ]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_input_ids = input_ids[batch_idx]
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[0]
                cur_input_embeds_1 = input_embeds_1[batch_idx]
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx].unsqueeze(1).expand(-1, tokens.shape[-1]))
                continue

            cur_input_embeds = input_embeds_1[batch_idx]
            image_token_indices = (
                [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            )
            cur_labels = labels[batch_idx]

            cur_input_ids_noim = []
            cur_labels_noim = []
            cur_input_embeds_no_im = []

            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_input_embeds_no_im.append(cur_input_embeds[image_token_indices[i] + 1 : image_token_indices[i + 1]])

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i].unsqueeze(1).expand(-1, tokens.shape[-1]))
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_tokens = tokens[cur_image_idx]
                    cur_new_input_embeds.append(cur_image_features)
                    if self.config.mm_use_vi_start_end:
                        if (cur_input_ids[-3] == -200 and self.llm.vocab_size - 4 in cur_new_labels[-1]) \
                             or all(x == -200 for x in cur_input_ids[-10:-3]):
                            cur_new_labels.append(cur_tokens)
                        else:
                            cur_new_labels.append(
                                torch.full(
                                    (cur_image_features.shape[0], tokens.shape[-1]),
                                    IGNORE_INDEX,
                                    device=cur_labels.device,
                                    dtype=cur_labels.dtype,
                                )
                            )
                    else:
                        if (cur_input_ids[-3] == -200 and self.llm.vocab_size - 2 in cur_new_labels[-1]):
                            cur_new_labels.append(cur_tokens)
                        else:
                            cur_new_labels.append(
                                torch.full(
                                    (cur_image_features.shape[0], tokens.shape[-1]),
                                    IGNORE_INDEX,
                                    device=cur_labels.device,
                                    dtype=cur_labels.dtype,
                                )
                            )                        
                    cur_image_idx += 1

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.llm.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            if any(len(x) > tokenizer_model_max_length for x in new_input_embeds):
                warnings.warn("Inputs truncated!")
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len, tokens.shape[-1]),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.llm.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    def repack_multimodal_data(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        inputs_embeds,
        labels,
    ):
        new_inputs_embeds = []
        new_position_ids = []
        new_labels = []

        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        sorted_seqlens_in_batch, sorted_idx = torch.sort(seqlens_in_batch, descending=True)
        max_seqlen = inputs_embeds.shape[1]

        cur_inputs_embeds = []
        cur_position_ids = []
        cur_labels = []
        cur_batch_len = 0

        for i in range(len(sorted_seqlens_in_batch)):
            cur_seqlen = sorted_seqlens_in_batch[i].item()
            if cur_seqlen + cur_batch_len <= max_seqlen:
                cur_batch_len += cur_seqlen
                cur_inputs_embeds.append(inputs_embeds[sorted_idx[i]][attention_mask[sorted_idx[i]]])
                cur_position_ids.append(
                    torch.arange(
                        cur_inputs_embeds[-1].shape[0],
                        device=cur_inputs_embeds[-1].device,
                    )
                )
                cur_labels.append(labels[sorted_idx[i]][attention_mask[sorted_idx[i]]])
            else:
                new_inputs_embeds.append(torch.cat(cur_inputs_embeds, 0))
                new_position_ids.append(torch.cat(cur_position_ids, 0))
                new_labels.append(torch.cat(cur_labels, 0))

                cur_batch_len = cur_seqlen
                cur_inputs_embeds = [inputs_embeds[sorted_idx[i]][attention_mask[sorted_idx[i]]]]
                cur_position_ids = [
                    torch.arange(
                        cur_inputs_embeds[-1].shape[0],
                        device=cur_inputs_embeds[-1].device,
                    )
                ]
                cur_labels = [labels[sorted_idx[i]][attention_mask[sorted_idx[i]]]]

        if len(cur_inputs_embeds):
            new_inputs_embeds.append(torch.cat(cur_inputs_embeds, 0))
            new_position_ids.append(torch.cat(cur_position_ids, 0))
            new_labels.append(torch.cat(cur_labels, 0))

        new_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            new_inputs_embeds, batch_first=True, padding_value=self.llm.pad_token_id
        )
        new_position_ids = torch.nn.utils.rnn.pad_sequence(new_position_ids, batch_first=True, padding_value=-1)
        new_labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_INDEX)
        new_attention_mask = new_position_ids.ne(-1)
        assert new_attention_mask.sum() == attention_mask.sum()

        return (
            None,
            new_position_ids,
            new_attention_mask,
            past_key_values,
            new_inputs_embeds,
            new_labels,
            sorted_seqlens_in_batch,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            if model_args.mm_use_vi_start_end:
                num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN], special_tokens=True)
            else:
                num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        cfg: Optional[float] = 3.0,
        **generation_kwargs,
    ):
        if images is not None:
            (_, _, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                input_ids, None, attention_mask, None, None, images
            )
        else:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds.to(self.dtype)

        if images is not None:
            outputs = self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generation_kwargs)

            return outputs
        else:
            image_ids = []
            outputs = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_tower=self.vision_tower,
                mm_projector=self.mm_projector,
                image_ids=image_ids,
                cfg=cfg,
                **generation_kwargs
            )
            image_ids = torch.cat(image_ids, dim=1)

            return image_ids
    
    @torch.inference_mode()
    def generate_content(self, prompt: Union[str, List], generation_config: Optional[GenerationConfig] = None) -> str:
        conversation = [{"from": "human", "value": prompt}]

        media = extract_media(conversation, self.config)

        if "image" in media:
            images = process_images(media["image"], self.vision_tower.image_processor, self.config).to(self.device, dtype=eval(self.config.model_dtype))
        else:
            images = None

        input_ids = tokenize_conversation(conversation, self.tokenizer, add_generation_prompt=True).cuda().unsqueeze(0)

        generation_config = generation_config or self.default_generation_config

        output_ids = self.generate(input_ids=input_ids, images=images, generation_config=generation_config)

        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        return response
    
    @torch.inference_mode()
    def generate_image_content(self, prompt: str, cfg: float = 3.0, generation_nums: int = 1) -> torch.Tensor:
        input_ids_list = []

        conversation = [{"from": "human", "value": prompt}]
        input_ids = tokenize_conversation(conversation, self.tokenizer, add_generation_prompt=True, image_generation=True).cuda()
        input_ids_list += [input_ids] * generation_nums

        cfg_conversation = [{"from": "human", "value": " "}]
        cfg_input_ids = tokenize_conversation(cfg_conversation, self.tokenizer, add_generation_prompt=True, image_generation=True).cuda()
        input_ids_list += [cfg_input_ids] * generation_nums

        max_length = max([len(input_ids) for input_ids in input_ids_list])
        input_ids = torch.zeros((len(input_ids_list), max_length), dtype=input_ids_list[0].dtype).cuda()
        attention_mask = torch.zeros((len(input_ids_list), max_length)).bool().cuda()
        for i in range(len(input_ids_list)):
            input_ids[i, -len(input_ids_list[i]):] = input_ids_list[i]
            attention_mask[i, -len(input_ids_list[i]):] = True

        image_ids = self.generate(input_ids=input_ids, attention_mask=attention_mask, cfg=cfg, max_new_tokens=self.vision_tower.image_tokens, use_cache=True)

        image_embeds = self.vision_tower.vision_tower.rqtransformer.embed_with_model_aux(image_ids, self.vision_tower.vision_tower.rqvaesiglip)
        image_embeds = torch.cumsum(image_embeds, dim=-2)[:,:,-1,:]
        image_embeds = image_embeds.reshape(input_ids.shape[0], int(self.vision_tower.image_tokens**0.5), int(self.vision_tower.image_tokens**0.5), -1)
        response = self.vision_tower.vision_tower.rqvaesiglip.decode(image_embeds).to(torch.float32).add_(1).mul_(127.5).clamp_(0, 255)

        return response.chunk(2)[0]
    
    @torch.inference_mode()
    def generate_video_content(self, prompt: str, cfg: float = 3.0, generation_nums: int = 1) -> torch.Tensor:
        GENERATION_VIDEO_FRAMES = 8
        input_ids_list = []

        conversation = [{"from": "human", "value": prompt}]
        input_ids = tokenize_conversation(conversation, self.tokenizer, add_generation_prompt=True, video_generation=True).cuda()
        input_ids_list += [input_ids] * generation_nums

        cfg_conversation = [{"from": "human", "value": " "}]
        cfg_input_ids = tokenize_conversation(cfg_conversation, self.tokenizer, add_generation_prompt=True, video_generation=True).cuda()
        input_ids_list += [cfg_input_ids] * generation_nums

        max_length = max([len(input_ids) for input_ids in input_ids_list])
        input_ids = torch.zeros((len(input_ids_list), max_length), dtype=input_ids_list[0].dtype).cuda()
        attention_mask = torch.zeros((len(input_ids_list), max_length)).bool().cuda()
        for i in range(len(input_ids_list)):
            input_ids[i, -len(input_ids_list[i]):] = input_ids_list[i]
            attention_mask[i, -len(input_ids_list[i]):] = True

        video_ids = self.generate(input_ids=input_ids, attention_mask=attention_mask, cfg=cfg, max_new_tokens=self.vision_tower.image_tokens * GENERATION_VIDEO_FRAMES, use_cache=True)

        video_embeds = self.vision_tower.vision_tower.rqtransformer.embed_with_model_aux(video_ids, self.vision_tower.vision_tower.rqvaesiglip)
        video_embeds = torch.cumsum(video_embeds, dim=-2)[:,:,-1,:]
        video_embeds = video_embeds.reshape(input_ids.shape[0] * GENERATION_VIDEO_FRAMES, int(self.vision_tower.image_tokens**0.5), int(self.vision_tower.image_tokens**0.5), -1)
        response = self.vision_tower.vision_tower.rqvaesiglip.decode(video_embeds).to(torch.float32).add_(1).mul_(127.5).clamp_(0, 255)
        _, _, H, W = response.shape
        response = response.reshape(input_ids.shape[0], GENERATION_VIDEO_FRAMES, 3, H, W)

        return response.chunk(2)[0]

    @property
    def default_generation_config(self) -> GenerationConfig:
        generation_config = copy.deepcopy(self.llm.generation_config)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token")
        if generation_config.max_length == GenerationConfig().max_length:
            generation_config.max_length = self.tokenizer.model_max_length
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if generation_config.bos_token_id is None:
            generation_config.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = self.tokenizer.convert_tokens_to_ids(infer_stop_tokens(self.tokenizer))
        return generation_config
