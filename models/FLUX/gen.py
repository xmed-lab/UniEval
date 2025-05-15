import os
import torch

from optimum.quanto import freeze, qfloat8, quantize

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast


# int8 models to avoid out of memory
class FLUXD:

    def __init__(self, model_name):
        dtype = torch.bfloat16
        
        bfl_repo = model_name
        revision = "refs/pr/3"
        
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
        tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
        vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
        transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision)
        
        quantize(transformer, weights=qfloat8)
        freeze(transformer)
        
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)
        
        self.pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=None,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=None,
        )
        self.pipe.text_encoder_2 = text_encoder_2
        self.pipe.transformer = transformer
        self.pipe.enable_model_cpu_offload()

    def generate(self, input_text, img_num, save_dir):
        # avoid out of memory, b=1
        os.makedirs(save_dir, exist_ok=True)
        outputs = []
        for i in range(img_num):
            image = self.pipe(
                input_text,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]

            path = os.path.join(save_dir, f"image_{i}.png")
            image.save(path)
            outputs.append(path)

        return outputs


class FLUXS:

    def __init__(self, model_name):
        dtype = torch.bfloat16
        
        bfl_repo = model_name
        revision = "refs/pr/1"
        
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
        tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
        vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
        transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision)
        
        quantize(transformer, weights=qfloat8)
        freeze(transformer)
        
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)
        
        self.pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=None,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=None,
        )
        self.pipe.text_encoder_2 = text_encoder_2
        self.pipe.transformer = transformer
        self.pipe.enable_model_cpu_offload()

    def generate(self, input_text, img_num, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        outputs = []
        for i in range(img_num):
            image = self.pipe(
                prompt=input_text,
                height=1024,
                width=1024,
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(i)
            ).images[0]

            path = os.path.join(save_dir, f"image_{i}.png")
            image.save(path)
            outputs.append(path)

        return outputs

#class FLUXD:
#
#    def __init__(self, model_name):
#        self.pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
#        torch.cuda.empty_cache()
#        self.pipe.enable_attention_slicing()
#        self.pipe.enable_sequential_cpu_offload()
#
#    def generate(self, input_text, img_num, save_dir):
#        # avoid out of memory, b=1
#        os.makedirs(save_dir, exist_ok=True)
#        outputs = []
#        for i in range(img_num):
#            image = self.pipe(
#                input_text,
#                height=1024,
#                width=1024,
#                guidance_scale=3.5,
#                num_inference_steps=50,
#                max_sequence_length=512,
#                generator=torch.Generator("cpu").manual_seed(0)
#            ).images[0]
#            
#            path = os.path.join(save_dir, f"image_{i}.png")
#            image.save(path)
#            outputs.append(path)
#
#        return outputs
#
#
#class FLUXS:
#
#    def __init__(self, model_name):
#        self.pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
#        torch.cuda.empty_cache()
#        self.pipe.enable_attention_slicing()
#        self.pipe.enable_sequential_cpu_offload()
#        #self.pipe.enable_model_cpu_offload()
#
#    def generate(self, input_text, img_num, save_dir):
#        os.makedirs(save_dir, exist_ok=True)
#        outputs = []
#        for i in range(img_num):
#            image = self.pipe(
#                input_text,
#                height=1024,
#                width=1024,
#                guidance_scale=0.0,
#                num_inference_steps=4,
#                max_sequence_length=256,
#                generator=torch.Generator("cpu").manual_seed(i)
#            ).images[0]
#
#            path = os.path.join(save_dir, f"image_{i}.png")
#            image.save(path)
#            outputs.append(path)
#
#        return outputs
#
