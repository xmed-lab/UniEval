import argparse
import numpy as np
import os
import torch
import string
from PIL import Image
import random
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from tqdm import tqdm
from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, create_attention_mask_for_mmu
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer
import torch.nn.functional as F

def save_image(response, input_text="", path='generated_samples/showo'):
    os.makedirs(path, exist_ok=True)
    outputs = []
    trans = str.maketrans(' ', '-', string.punctuation)
    save_prompt = input_text.translate(trans)
    for i in range(len(response)):
        response[i].save(os.path.join(path, f"image_{i}.png"))
        outputs.append(os.path.join(path, f"image_{i}.png"))
    return outputs


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

class Show_o:
    def __init__(self):
        torch.set_default_tensor_type(torch.FloatTensor)

        self.config = get_config()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.showo.llm_model_path, padding_side="left")

        self.uni_prompting = UniversalPrompting(tokenizer, max_text_len=self.config.dataset.preprocessing.max_seq_length,
                                           special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>",
                                           "<|v2v|>", "<|lvg|>"),
                                           ignore_id=-100, cond_dropout_prob=self.config.training.cond_dropout_prob)

        self.vq_model = get_vq_model_class(self.config.model.vq_model.type)
        self.vq_model = self.vq_model.from_pretrained(self.config.model.vq_model.vq_model_name).to(self.device)
        self.vq_model.requires_grad_(False)
        self.vq_model.eval()



        self.model = Showo.from_pretrained(self.config.model.showo.pretrained_model_path).to(self.device)
        self.model.eval()

        # dtype = torch.bfloat16
        # self.model = self.model.to(dtype=dtype)
        # self.vq_model = self.vq_model.to(dtype=dtype)

        # vision_tower_name = "openai/clip-vit-large-patch14-336"
        # vision_tower = CLIPVisionTower(vision_tower_name).to(device)
        # clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)


        # load from users passed arguments
        if self.config.get("validation_prompts_file", None) is not None:
            self.config.dataset.params.validation_prompts_file = self.config.validation_prompts_file
        self.config.training.batch_size = self.config.batch_size
        self.config.training.guidance_scale = self.config.guidance_scale
        self.config.training.generation_timesteps = self.config.generation_timesteps


    def generate(self, input_text, img_num, save_dir):

        mask_token_id = self.model.config.mask_token_id

        prompts = [input_text for _ in range(img_num)]

        image_tokens = torch.ones((len(prompts), self.config.model.showo.num_vq_tokens),
                                  dtype=torch.long, device=self.device) * mask_token_id

        input_ids, _ = self.uni_prompting((prompts, image_tokens), 't2i_gen')

        if self.config.training.guidance_scale > 0:
            uncond_input_ids, _ = self.uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
            attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True)
        else:
            attention_mask = create_attention_mask_predict_next(input_ids,
                                                                pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True)
            uncond_input_ids = None

        if self.config.get("mask_schedule", None) is not None:
            schedule = self.config.mask_schedule.schedule
            args = self.config.mask_schedule.get("params", {})
            mask_schedule = get_mask_chedule(schedule, **args)
        else:
            mask_schedule = get_mask_chedule(self.config.training.get("mask_schedule", "cosine"))

        with torch.no_grad():
            gen_token_ids = self.model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                guidance_scale=self.config.training.guidance_scale,
                temperature=self.config.training.get("generation_temperature", 1.0),
                timesteps=self.config.training.generation_timesteps,
                noise_schedule=mask_schedule,
                noise_type=self.config.training.get("noise_type", "mask"),
                seq_len=self.config.model.showo.num_vq_tokens,
                uni_prompting=self.uni_prompting,
                config=self.config,
            )

        gen_token_ids = torch.clamp(gen_token_ids, max=self.config.model.showo.codebook_size - 1, min=0)
        images = self.vq_model.decode_code(gen_token_ids)

        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]

        outputs = save_image(pil_images, input_text, save_dir)

        return outputs


    def understand(self, input_img, prompt):

        # print(f"\033[91m {prompt} \033[0m")

        # q1 = f"<|image|>{prompt}"

        image_ori = Image.open(input_img).convert('RGB')
        image = image_transform(image_ori, resolution=self.config.dataset.params.resolution).to(self.device)
        image = image.unsqueeze(0)

        # pixel_values = clip_image_processor.preprocess(image_ori, return_tensors="pt")["pixel_values"][0]

        image_tokens = self.vq_model.get_code(image) + len(self.uni_prompting.text_tokenizer)


        input_ids = self.uni_prompting.text_tokenizer(['USER: \n' + prompt + ' ASSISTANT:'])[
            'input_ids']
        input_ids = torch.tensor(input_ids).to(self.device)

        input_ids = torch.cat([
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|mmu|>']).to(self.device),
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|soi|>']).to(self.device),
            image_tokens,
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|eoi|>']).to(self.device),
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|sot|>']).to(self.device),
            input_ids
        ], dim=1).long()

        attention_mask = create_attention_mask_for_mmu(input_ids.to(self.device),
                                                       eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']))

        cont_toks_list = self.model.mmu_generate(input_ids, attention_mask=attention_mask,
                                            max_new_tokens=self.config.max_new_tokens, top_k=1,
                                            eot_token=self.uni_prompting.sptids_dict['<|eot|>'])

        cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]

        text = self.uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)[0].replace(' ', '')


        return f"({text})"



