import os, torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline, StableDiffusion3Pipeline


class SDV15:

    def __init__(self, model_name):
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name, use_safetensors=True)
        self.pipe.to('cuda')

    def generate(self, input_text, img_num, save_dir):
        images = self.pipe([input_text] * img_num).images
        os.makedirs(save_dir, exist_ok=True)
        outputs = []

        for i in range(len(images)):
            path = os.path.join(save_dir, f"image_{i}.png")
            images[i].save(path)
            outputs.append(path)

        return outputs


class SDV21:
    
    def __init__(self, model_name):
        # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to('cuda')

    def generate(self, input_text, img_num, save_dir):
        images = self.pipe([input_text] * img_num).images
        os.makedirs(save_dir, exist_ok=True)
        outputs = []

        for i in range(len(images)):
            path = os.path.join(save_dir, f"image_{i}.png")
            images[i].save(path)
            outputs.append(path)

        return outputs


class SDXL:

    def __init__(self, model_name):
        self.pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, \
            use_safetensors=True, variant="fp16")
        self.pipe.to('cuda')

    def generate(self, input_text, img_num, save_dir):
        images = self.pipe([input_text] * img_num).images
        os.makedirs(save_dir, exist_ok=True)
        outputs = []

        for i in range(len(images)):
            path = os.path.join(save_dir, f"image_{i}.png")
            images[i].save(path)
            outputs.append(path)

        return outputs


class SD3M:

    def __init__(self, model_name):
        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self.pipe.enable_model_cpu_offload()
        #self.pipe.to('cuda')

    def generate(self, input_text, img_num, save_dir):
        images = self.pipe(
            [input_text] * img_num,
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=7.0,
        ).images

        os.makedirs(save_dir, exist_ok=True)
        outputs = []

        for i in range(len(images)):
            path = os.path.join(save_dir, f"image_{i}.png")
            images[i].save(path)
            outputs.append(path)
        
        return outputs


class SD35M:

    def __init__(self, model_name):
        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self.pipe.enable_model_cpu_offload()

    def generate(self, input_text, img_num, save_dir):
        images = self.pipe(
            [input_text] * img_num,
            num_inference_steps=40,
            guidance_scale=4.5,
        ).images

        os.makedirs(save_dir, exist_ok=True)
        outputs = []

        for i in range(len(images)):
            path = os.path.join(save_dir, f"image_{i}.png")
            images[i].save(path)
            outputs.append(path)

        return outputs
