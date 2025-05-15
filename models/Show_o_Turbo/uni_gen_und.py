import argparse
import numpy as np
import os
import torch
import string
from PIL import Image
import random
from tqdm import tqdm
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config, flatten_omega_conf, image_transform
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from models import Showo, MAGVITv2, get_mask_chedule
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer
from omegaconf import OmegaConf


def create_attention_mask_for_mmu(sequence, eoi_id=128258, return_inverse_mask=True):
    N, L = sequence.shape
    causal_mask = torch.tril(torch.ones((N, 1, L, L), dtype=torch.bool)).to(sequence.device)
    eoi_image = torch.where(sequence == eoi_id)[1]
    causal_mask[:, :, :, :eoi_image[0] + 1] = 1

    if return_inverse_mask:
        inverted_mask = 1.0 - causal_mask.type(sequence.dtype)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(sequence.dtype).min
        )
        return inverted_mask
    else:
        return causal_mask



def clean_prompt(prompt):
    # Remove line breaks
    cleaned_prompt = prompt.replace('\n', ' ')
    # Remove <image> tag
    cleaned_prompt = cleaned_prompt.replace('<image>', '')
    return cleaned_prompt


####### Get jacobian trajectory #######
@torch.inference_mode()
def get_jacobian_trajectory(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens,
    device,
    uni_prompting
    ):
    """
    Generates text using Jacobian trajectory sampling.

    Args:
        model: Show-o model.
        tokenizer: Tokenizer.
        input_ids (torch.Tensor): Input token IDs.
        attention_mask (torch.Tensor): Attention mask.
        max_new_tokens (int): Maximum new tokens to generate per iteration.

    Returns:
        tuple: Trajectory IDs, last logits trajectory, eos_reached flag, iteration count.
    """

    bsz = input_ids.shape[0]
    prompt_len = [input_ids[i].shape[0] for i in range(bsz)]
    max_prompt_len = max(prompt_len)
    total_len = max_prompt_len + max_new_tokens

    # initialize the first point of jacobian trajectory
    tokens = torch.full((bsz, total_len), tokenizer.pad_token_id, dtype=torch.long, device=device)

    for i in range(bsz):
        max_index = len(uni_prompting.text_tokenizer) - 1
        filtered_choices = [x for x in input_ids[i] if 0 <= x <= max_index]
        tokens[i, :] = torch.tensor(random.choices(filtered_choices, k=total_len)).to(dtype=torch.long, device="cuda")
        tokens[i, : prompt_len[i]] = torch.tensor(input_ids[i][: prompt_len[i]], dtype=torch.long, device="cuda")

    trajectory = []
    logits_trajectory = []
    next_generation = tokens
    next_text = tokenizer.decode(next_generation.tolist()[0])

    generate_attention_mask = create_attention_mask_for_mmu(next_generation.to(device),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
    trajectory.append(tokens)
    itr = 0

    while True:
        current_generation = next_generation
        logits = model(current_generation, attention_mask=generate_attention_mask)

        logits_trajectory.append(logits)
        next_generation = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.01, dim=-1)

        # keep prompt unchanged, update generated tokens
        for i in range(bsz):
            next_generation[i, :] = torch.cat((tokens[i, :prompt_len[i]], next_generation[i, prompt_len[i] - 1:total_len - 1]), dim=0)

        trajectory.append(next_generation)
        itr += 1

        if torch.all(torch.eq(next_generation, current_generation)).item():
            eos_idxs = torch.where(trajectory[-1][0] == tokenizer.eos_token_id)
            eos_reached = len(eos_idxs[0]) > 1
            return trajectory[:-1], logits_trajectory[-1], eos_reached, itr



def forward(
    model,
    input_ids,
    input_embeddings=None,
    attention_mask=None,
    labels=None,
    label_smoothing=0.0,
    config=None,
    labels_mask_text=None,
    labels_mask_image=None,
    **kwargs,
):
    attention_mask = attention_mask.to(dtype=model.dtype)
    if input_embeddings is None:
        logits = model.showo(input_ids=input_ids, attention_mask=attention_mask)['logits']
    else:
        logits = model.showo(inputs_embeds=input_embeddings, attention_mask=attention_mask)['logits']

    if labels is not None:
        raise NotImplementedError

    return logits

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t, generator=None):
    noise = torch.zeros_like(t).uniform_(0, 1, generator=generator)
    return -log(-log(noise))


def mask_by_random_topk(mask_len, probs, temperature=1.0, generator=None):
    confidence = log(probs) + temperature * gumbel_noise(probs, generator=generator)
    sorted_confidence = torch.sort(confidence, dim=-1).values
    cut_off = torch.gather(sorted_confidence, 1, mask_len.long())
    masking = confidence < cut_off
    return masking


def denoise(model, input_ids, input_ids_minus_lm_vocab_size, uncond_input_ids, uncond_prefix, attention_mask, config,
            generator, ratio, mask_token_id, noise_schedule, seq_len, temperature, top_k):
    if uncond_input_ids is not None and config.training.guidance_scale > 0:
        uncond_input_ids = torch.cat(
            [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
        model_input = torch.cat([input_ids, uncond_input_ids])
        cond_logits, uncond_logits = forward(model, model_input, attention_mask=attention_mask).chunk(2)
        logits = (1 + config.training.guidance_scale) * cond_logits - config.training.guidance_scale * uncond_logits
        logits = logits[:, -(seq_len + 1):-1, config.model.showo.llm_vocab_size + 10:-1]
    else:
        logits = forward(model, input_ids, attention_mask=attention_mask)
        logits = logits[:, -(seq_len + 1):-1, config.model.showo.llm_vocab_size + 10:-1]

    probs = logits.softmax(dim=-1)
    sampled = probs.reshape(-1, logits.size(-1))

    if top_k is not None:
        topk_probs, topk_indices = torch.topk(sampled, top_k, dim=-1)
        topk_probs /= topk_probs.sum(dim=-1, keepdim=True)
        sampled_ids = torch.multinomial(topk_probs, 1, generator=generator)[:, 0]
        sampled_ids = topk_indices.gather(-1, sampled_ids.view(-1, 1)).view(*logits.shape[:-1])

    else:
        sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

    unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
    sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)

    mask_ratio = noise_schedule(torch.tensor(ratio))
    selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
    selected_probs = selected_probs.squeeze(-1)
    selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)

    mask_len = (seq_len * mask_ratio).floor().unsqueeze(0).to(logits.device)
    mask_len = torch.max(
        torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
    )
    temperature = temperature * (1.0 - ratio)
    masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)

    input_ids[:, -(seq_len + 1):-1] = torch.where(masking, mask_token_id,
                                                  sampled_ids + config.model.showo.llm_vocab_size + 10)
    input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

    return input_ids, input_ids_minus_lm_vocab_size, temperature, sampled_ids


def save_image(response, input_text="", path='generated_samples/showoturbo'):
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

class Show_o_Turbo:
    def __init__(self, config_file_path="models/Show_o_Turbo/config/showo_256.yaml", inference_step=16, guidance_scale=0):

        self.config = OmegaConf.load(config_file_path)
        self.config.mode = 't2i'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.showo.llm_model_path, padding_side="left")

        self.uni_prompting = UniversalPrompting(self.tokenizer, max_text_len=self.config.dataset.preprocessing.max_seq_length,
                                           special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>",
                                           "<|v2v|>", "<|lvg|>"),
                                           ignore_id=-100, cond_dropout_prob=self.config.training.cond_dropout_prob)




        self.vq_model = get_vq_model_class(self.config.model.vq_model.type)
        self.vq_model = self.vq_model.from_pretrained(self.config.model.vq_model.vq_model_name).to(self.device)
        self.vq_model.requires_grad_(False)
        self.vq_model.eval()

        self.model = Showo.from_pretrained("SJTU-Deng-Lab/Show-o-Turbo-256").to(self.device)
        self.model.eval()


        dtype = torch.float16
        # self.model = self.model.to(dtype=dtype)
        # self.vq_model = self.vq_model.to(dtype=dtype)


        vision_tower_name = "openai/clip-vit-large-patch14-336"
        # vision_tower = CLIPVisionTower(vision_tower_name).to(device)
        # clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)


        # load from users passed arguments
        if self.config.get("validation_prompts_file", None) is not None:
            self.config.dataset.params.validation_prompts_file = self.config.validation_prompts_file
        self.config.training.batch_size = 1
        self.config.training.guidance_scale = guidance_scale
        self.config.training.generation_timesteps = inference_step


    def generate(self, input_text, img_num, save_dir):

        mask_token_id = self.model.config.mask_token_id
        top_k = 200

        prompts = [input_text for _ in range(img_num)]

        image_tokens = torch.ones((len(prompts), self.config.model.showo.num_vq_tokens),
                                  dtype=torch.long, device=self.device) * mask_token_id

        input_ids, _ = self.uni_prompting((prompts, image_tokens), 't2i_gen')

        if self.config.training.guidance_scale > 0:
            uncond_input_ids, _ = self.uni_prompting(([''] * 1, image_tokens), 't2i_gen')
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
            seed = 0  # fix seed for reproducibility
            generator = torch.Generator(device=self.device).manual_seed(seed)
            temperature = self.config.training.get("generation_temperature", 1.0)
            noise_schedule = mask_schedule
            noise_type = self.config.training.get("noise_type", "mask")
            seq_len = self.config.model.showo.num_vq_tokens

            input_ids_minus_lm_vocab_size = input_ids[:, -(seq_len + 1):-1].clone()
            input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                                        mask_token_id,
                                                        input_ids_minus_lm_vocab_size - self.config.model.showo.llm_vocab_size - 10)

            if uncond_input_ids is not None:
                uncond_prefix = uncond_input_ids[:, :self.config.dataset.preprocessing.max_seq_length + 1]
            else:
                uncond_prefix = None

            for step in range(self.config.training.generation_timesteps):
                ratio = 1.0 * (step + 1) / self.config.training.generation_timesteps
                input_ids, input_ids_minus_lm_vocab_size, temperature, sampled_ids = denoise(
                    self.model, input_ids, input_ids_minus_lm_vocab_size,
                    uncond_input_ids, uncond_prefix, attention_mask, self.config,
                    generator, ratio, mask_token_id, noise_schedule, seq_len, temperature, top_k)

        gen_token_ids = sampled_ids
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
        max_new_tokens = 16
        max_new_seq_len = 512


        image_ori = Image.open(input_img).convert('RGB')
        image = image_transform(image_ori, resolution=self.config.dataset.params.resolution).to(self.device)
        image = image.unsqueeze(0)


        image_tokens = self.vq_model.get_code(image) + len(self.uni_prompting.text_tokenizer)
        batch_size = 1

        prompt = clean_prompt(prompt)
        # print(f"\033[91m {prompt} \033[0m")
        input_text = ['USER: \n' + prompt + ' ASSISTANT:']

        input_ids = self.uni_prompting.text_tokenizer(input_text)['input_ids']
        input_ids = torch.tensor(input_ids).to(self.device)
        input_ids = torch.cat([
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|mmu|>']).to(self.device),
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|soi|>']).to(self.device),
            image_tokens,
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|eoi|>']).to(self.device),
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|sot|>']).to(self.device),
            input_ids
        ], dim=1).long()

        inputs = torch.Tensor(input_ids).to(device=self.model.device, dtype=torch.int)

        attention_mask = create_attention_mask_for_mmu(inputs.to(self.device),
                                                       eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']))

        itr = 0
        eos_reached = False
        jacobian_trajectory_ids = None  # Initialize to None for scope

        while itr * max_new_tokens < max_new_seq_len and not eos_reached:
            # print('Retrieving one Jacobian trajectory...')
            jacobian_trajectory_ids, teacher_logits, eos_reached, iitr = get_jacobian_trajectory(self.model, self.tokenizer,
                                                                                                 inputs, attention_mask,
                                                                                                 max_new_tokens, self.device, self.uni_prompting)


            itr += 1
            # print(f'Jacobi iteration: {itr}')
            inputs = jacobian_trajectory_ids[-1]
            if eos_reached:
                # print("EOS reached.")
                break
        # print(f"\033[92m {jacobian_trajectory_ids[-1][0].shape} \033[0m")
        # print(f"\033[92m {input_ids[0].shape[0]} \033[0m")

        answer = jacobian_trajectory_ids[-1][0][input_ids[0].shape[0]:].tolist()
        first_eos_index = next((i for i, token in enumerate(answer) if token == self.tokenizer.eos_token_id),
                               len(answer))
        fil_answer = answer[:first_eos_index]
        generated_text = self.tokenizer.decode(fil_answer).replace(' ', '')

        # print(f"\033[92m {generated_text} \033[0m")



        return f"({generated_text})"



