import argparse
import numpy as np
import os
import torch
import string
from PIL import Image
import requests
import sys
from transformers import AutoProcessor, AutoTokenizer
from vargpt_llava.modeling_vargpt_llava import VARGPTLlavaForConditionalGeneration
from vargpt_llava.prepare_vargpt_llava import prepare_vargpt_llava
from vargpt_llava.processing_vargpt_llava import VARGPTLlavaProcessor
from patching_utils.patching import patching



class VARGPT:
    def __init__(self, model_id='VARGPT-family/VARGPT_LLaVA-v1'):
        torch.set_default_tensor_type('torch.HalfTensor')

        prepare_vargpt_llava(model_id)

        self.model = VARGPTLlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(0)

        patching(self.model)
        self.processor = VARGPTLlavaProcessor.from_pretrained(model_id)

    def generate(self, input_text, img_num, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        outputs = []
        trans = str.maketrans(' ', '-', string.punctuation)
        save_prompt = input_text.translate(trans)


        for i in range(img_num):
            outputs.append(os.path.join(save_dir, f"image_{i}.png"))

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Please design a drawing of {input_text}"},
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

            inputs = self.processor(text=prompt, return_tensors='pt').to(0, torch.float32)
            self.model._IMAGE_GEN_PATH = os.path.join(save_dir, f"image_{i}.png")
            output = self.model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=False)

        return outputs

    def understand(self, input_img, prompt):

        # print(f"\033[91m {prompt} \033[0m")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        try:
            raw_image = Image.open(input_img)
        except:
            return ''

        inputs = self.processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.bfloat16)

        output = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False)

        conv = self.processor.decode(output[0], skip_special_tokens=True)
        ans = conv.split('ASSISTANT: ')[-1]
        # print(f"\033[92m {ans} \033[0m")
        return f"({ans})"



