import os
import torch

from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..configuration_vila_u import VILAUConfig
from ..vila_u_arch import VILAUMetaModel, VILAUMetaForCausalLM


class VILAULlamaConfig(VILAUConfig):
    model_type = "vila_u_llama"


class VILAULlamaModel(VILAUMetaModel, VILAUMetaForCausalLM, PreTrainedModel):
    config_class = VILAULlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: VILAULlamaConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        
        return self.init_vlm(config=config, *args, **kwargs)
        
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token, 
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )

        return super(VILAULlamaModel).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token, 
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )    

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
            )
            
        if self.training:
            (
                _,
                new_position_ids,
                new_attention_mask,
                _,
                new_inputs_embeds,
                new_labels,
                sorted_seqlens_in_batch,
            ) = self.repack_multimodal_data(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            )
            new_input_ids = None
            past_key_values = None
        else:
            new_attention_mask = attention_mask
            new_position_ids = position_ids
            new_inputs_embeds = inputs_embeds
            new_labels = labels
            sorted_seqlens_in_batch = attention_mask.sum(-1).int()
            new_input_ids = input_ids

        output_attentions = output_attentions if output_attentions is not None else self.llm.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.llm.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.llm.config.use_return_dict

        outputs = self.llm.model(
            input_ids=new_input_ids,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=new_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            seqlens_in_batch=sorted_seqlens_in_batch,
        )

        hidden_states = outputs[0]

        image_hidden_states = []
        image_labels = []
        noimage_labels = []

        for i in range(hidden_states.shape[0]):
            label = new_labels[i]
            hidden_state = hidden_states[i]
            label_zero = label[:, 0].clone()

            if self.config.mm_use_vi_start_end:
                image_start_index = torch.nonzero(torch.eq(label_zero, self.llm.vocab_size - 4)).squeeze(1)
                image_end_index = torch.nonzero(torch.eq(label_zero, self.llm.vocab_size - 3)).squeeze(1)
                video_start_index = torch.nonzero(torch.eq(label_zero, self.llm.vocab_size - 2)).squeeze(1)
                video_end_index = torch.nonzero(torch.eq(label_zero, self.llm.vocab_size - 1)).squeeze(1)
                image_start_index = torch.cat([image_start_index, video_start_index])
                image_end_index = torch.cat([image_end_index, video_end_index])
            else:
                image_start_index = torch.nonzero(torch.eq(label_zero, self.llm.vocab_size - 2)).squeeze(1)
                image_end_index = torch.nonzero(torch.eq(label_zero, self.llm.vocab_size - 1)).squeeze(1)

            assert len(image_start_index) == len(image_end_index), f"length of image_start_index is {len(image_start_index)}, length of image_end_index is {len(image_end_index)}"

            if len(image_start_index) > 0:
                for start_idx, end_idx in zip(image_start_index, image_end_index):
                    image_label = label[start_idx+1:end_idx, :]
                    image_labels.append(image_label)
                    image_hidden_state = hidden_state[start_idx:end_idx-1, :]
                    image_hidden_states.append(image_hidden_state)
                    label_zero[start_idx+1:end_idx] = -100

            noimage_labels.append(label_zero)
        
        # For video
        image_hidden_states_aux = []
        image_labels_aux = []
        image_hidden_states_length = [img.shape[0] for img in image_hidden_states]
        image_hidden_states_length_relative = [img // min(image_hidden_states_length) for img in image_hidden_states_length]
        for l in range(len(image_hidden_states_length_relative)):
            if image_hidden_states_length_relative[l] > 1:
                image_hidden_states_aux += torch.split(image_hidden_states[l], min(image_hidden_states_length), dim=0)
                image_labels_aux += torch.split(image_labels[l], min(image_hidden_states_length), dim=0)
            else:
                image_hidden_states_aux.append(image_hidden_states[l])
                image_labels_aux.append(image_labels[l])

        if len(image_hidden_states_aux) > 0:
            image_hidden_states = torch.stack(image_hidden_states_aux, 0)
            image_labels = torch.stack(image_labels_aux, 0)

        noimage_labels = torch.stack(noimage_labels, 0)

        logits = self.llm.lm_head(hidden_states)

        loss_fct = CrossEntropyLoss()

        image_loss = None
        if torch.is_tensor(image_hidden_states):
            if hasattr(self.vision_tower.vision_tower, "rqvaesiglip"):
                outs = self.vision_tower.vision_tower.rqtransformer(image_hidden_states, image_labels - self.llm.vocab_size, self.vision_tower.vision_tower.rqvaesiglip)
            else:
                raise NotImplementedError()
            B, seq_len, D, C = outs.shape
            image_logits = outs.reshape(B*seq_len*D, C).contiguous()
            image_labels = image_labels.reshape(B*seq_len*D).contiguous() - self.llm.vocab_size
            image_loss = loss_fct(image_logits, image_labels)

        loss = None
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = noimage_labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, self.llm.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        if image_loss is not None:
            loss = loss + image_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        

AutoConfig.register("vila_u_llama", VILAULlamaConfig)
AutoModel.register(VILAULlamaConfig, VILAULlamaModel)