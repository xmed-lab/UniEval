import argparse
import cv2
import numpy as np
import os
import vila_u
import string

def save_image(response, input_text="", path='generated_samples/vilau'):
    os.makedirs(path, exist_ok=True)
    outputs = []
    trans = str.maketrans(' ', '-', string.punctuation)
    save_prompt = input_text.translate(trans)
    for i in range(response.shape[0]):
        image = response[i].permute(1, 2, 0)
        image = image.cpu().numpy().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path, f"image_{i}.png"), image)
        outputs.append(os.path.join(path, f"image_{i}.png"))

    return outputs


class VILAU:
    def __init__(self, model_path=''):
        self.model = vila_u.load(model_path)
        self.cfg = 3.0
        generation_config = self.model.default_generation_config
        generation_config.temperature = 0.9
        generation_config.top_p = 0.6

    def generate(self, input_text, img_num, save_dir):
        response = self.model.generate_image_content(input_text, self.cfg, img_num)
        outputs = save_image(response, input_text, save_dir)
        return outputs

    def understand(self, input_img, prompt):
        image = vila_u.Image(input_img)
        response = self.model.generate_content([image, prompt])
        return response



