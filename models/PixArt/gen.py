import os
import torch
from diffusers import PixArtAlphaPipeline


class PixArt:

    def __init__(self, model_name):
        self.pipe = PixArtAlphaPipeline.from_pretrained(model_name, torch_dtype=torch.float16, \
            use_safetensors=True, device_map="balanced")

    def generate(self, input_text, img_num, save_dir):
        images = self.pipe([input_text] * img_num).images

        os.makedirs(save_dir, exist_ok=True)
        outputs = []

        for i in range(len(images)):
            path = os.path.join(save_dir, f"image_{i}.png")
            images[i].save(path)
            outputs.append(path)

        return outputs
