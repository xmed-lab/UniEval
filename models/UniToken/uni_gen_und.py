import argparse
import numpy as np
import os
from unitoken.inference_solver_anyres import FlexARInferenceSolverAnyRes
import torch
import string
from PIL import Image


def save_image(response, input_text="", path='generated_samples/unitoken'):
    os.makedirs(path, exist_ok=True)
    outputs = []
    trans = str.maketrans(' ', '-', string.punctuation)
    save_prompt = input_text.translate(trans)
    for i in range(len(response)):
        a1, image = response[i][0], response[i][1][0]
        image.save(os.path.join(path, f"image_{i}.png"))
        outputs.append(os.path.join(path, f"image_{i}.png"))
    return outputs


class UniToken:
    def __init__(self, model_name='OceanJay/UniToken-AnyRes-StageII'):
        self.inference_solver = FlexARInferenceSolverAnyRes(
            model_path=model_name,
            precision="bf16",
            target_size=512,
        )

    def generate(self, input_text, img_num, save_dir):


        q1 = f"Generate an image according to the following prompt:\n" \
             f"{input_text}"

        # generated: tuple of (generated response, list of generated images)
        generated = self.inference_solver.generate_img(
            images=[],
            qas=[[q1, None]],
            max_gen_len=1536,
            temperature=1.0,
            num_return_sequences=img_num,
            logits_processor=self.inference_solver.create_logits_processor(cfg=3.0, image_top_k=4000),
        )


        # print(f"\033[92m {len(generated)} \033[0m")

        outputs = save_image(generated, input_text, save_dir)

        return outputs


    def understand(self, input_img, prompt):

        # print(f"\033[91m {prompt} \033[0m")

        q1 = f"<|image|>{prompt}"

        images = [Image.open(input_img).convert('RGB')]
        qas = [[q1, None]]

        # `len(images)` should be equal to the number of appearance of "<|image|>" in qas
        generated = self.inference_solver.generate(
            images=images,
            qas=qas,
            max_gen_len=512,
            temperature=1.0,
            logits_processor=self.inference_solver.create_logits_processor(cfg=4.0, image_top_k=2000),
        )

        a1 = generated[0]

        # print(f"\033[92m {a1} \033[0m")
        return a1



