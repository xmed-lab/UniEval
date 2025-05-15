# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import numpy as np
import PIL.Image

import torch
import torchvision
from transformers import AutoModelForCausalLM

from janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from diffusers.models import AutoencoderKL


@torch.inference_mode()
def generate(
    vl_gpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    vae,
    prompt: str,
    cfg_weight: float = 5.0,
    num_inference_steps: int = 30,
    batchsize: int = 5,
    save_dir: str = ''
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)
    
    tokens = torch.stack([input_ids] * 2 * batchsize).cuda()
    tokens[batchsize:, 1:] = vl_chat_processor.pad_id
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)

    # we remove the last <bog> token and replace it with t_emb later
    inputs_embeds = inputs_embeds[:, :-1, :] 
    
    # generate with rectified flow ode
    # step 1: encode with vision_gen_enc
    z = torch.randn((batchsize, 4, 48, 48), dtype=torch.bfloat16).cuda()
    
    dt = 1.0 / num_inference_steps
    dt = torch.zeros_like(z).cuda().to(torch.bfloat16) + dt
    
    # step 2: run ode
    attention_mask = torch.ones((2*batchsize, inputs_embeds.shape[1]+577)).to(vl_gpt.device)
    attention_mask[batchsize:, 1:inputs_embeds.shape[1]] = 0
    attention_mask = attention_mask.int()
    for step in range(num_inference_steps):
        # prepare inputs for the llm
        z_input = torch.cat([z, z], dim=0) # for cfg
        t = step / num_inference_steps * 1000.
        t = torch.tensor([t] * z_input.shape[0]).to(dt)
        z_enc = vl_gpt.vision_gen_enc_model(z_input, t)
        z_emb, t_emb, hs = z_enc[0], z_enc[1], z_enc[2]
        z_emb = z_emb.view(z_emb.shape[0], z_emb.shape[1], -1).permute(0, 2, 1)
        z_emb = vl_gpt.vision_gen_enc_aligner(z_emb)
        llm_emb = torch.cat([inputs_embeds, t_emb.unsqueeze(1), z_emb], dim=1)

        # input to the llm
        # we apply attention mask for CFG: 1 for tokens that are not masked, 0 for tokens that are masked.
        if step == 0:
            outputs = vl_gpt.language_model.model(inputs_embeds=llm_emb, 
                                             use_cache=True, 
                                             attention_mask=attention_mask,
                                             past_key_values=None)
            past_key_values = []
            for kv_cache in past_key_values:
                k, v = kv_cache[0], kv_cache[1]
                past_key_values.append((k[:, :, :inputs_embeds.shape[1], :], v[:, :, :inputs_embeds.shape[1], :]))
            past_key_values = tuple(past_key_values)
        else:
            outputs = vl_gpt.language_model.model(inputs_embeds=llm_emb, 
                                             use_cache=True, 
                                             attention_mask=attention_mask,
                                             past_key_values=past_key_values)
        hidden_states = outputs.last_hidden_state
        
        # transform hidden_states back to v
        hidden_states = vl_gpt.vision_gen_dec_aligner(vl_gpt.vision_gen_dec_aligner_norm(hidden_states[:, -576:, :]))
        hidden_states = hidden_states.reshape(z_emb.shape[0], 24, 24, 768).permute(0, 3, 1, 2)
        v = vl_gpt.vision_gen_dec_model(hidden_states, hs, t_emb)
        v_cond, v_uncond = torch.chunk(v, 2)
        v = cfg_weight * v_cond - (cfg_weight-1.) * v_uncond
        z = z + dt * v
        
    # step 3: decode with vision_gen_dec and sdxl vae
    dec = vae.decode(z / vae.config.scaling_factor).sample

    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    outputs = []
    os.makedirs(save_dir, exist_ok=True)
    for i in range(batchsize):
        save_path = os.path.join(save_dir, "img_{}.jpg".format(i))
        PIL.Image.fromarray(dec[i]).save(save_path)
        outputs.append(save_path)
    return outputs


class JanusFlow:
    def __init__(self, model_path='deepseek-ai/JanusFlow-1.3B'):
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        vl_gpt = MultiModalityCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        self.vae = self.vae.to(torch.bfloat16).cuda().eval()
        

    def generate(self, input_text, img_num, save_dir):
        conversation = [{"role": "User", "content": input_text,},
            {"role": "Assistant", "content": ""},]
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.vl_chat_processor.image_gen_tag

        return generate(self.vl_gpt, self.vl_chat_processor, self.vae, prompt, \
            cfg_weight=2.0, num_inference_steps=30, batchsize=img_num, save_dir=save_dir)


    def understand(self, input_img, prompt):
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder>\n" + prompt,
                "images": [input_img],
            },
            {"role": "Assistant", "content": ""},
        ]
        
        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)
        
        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        
        # run the model to get the response
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
        
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer
