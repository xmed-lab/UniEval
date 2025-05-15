import argparse
import torch
import numpy as np
from t2i.llava_t2i.dataset.process import crop_and_encode_text_and_img
from t2i.llava_t2i.utils import disable_torch_init
from tqdm import tqdm
import json
import gc
from PIL import Image
import os
from t2i.llava_t2i.model import *
import transformers
from transformers import TextStreamer
import string
import cv2
import i2t.llava as llava
from i2t.llava.model.builder import load_pretrained_model
from i2t.llava.conversation import conv_templates, SeparatorStyle
from i2t.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from i2t.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
import warnings

warnings.filterwarnings('ignore')

multi_step_infer_strategy = {
    1: {
        'topk_list': [600],
        'topp_list': [0.6],
    },
    2: {
        'topk_list': [1200, 1],
        'topp_list': [0.8, 0],
    },
    3: {
        'topk_list': [1200, 100, 1],
        'topp_list': [0.8, 0.8, 0],
    },
}



def save_image(response, input_text="", path='generated_samples/tokenflow'):
    os.makedirs(path, exist_ok=True)
    outputs = []
    trans = str.maketrans(' ', '-', string.punctuation)
    save_prompt = input_text.translate(trans)
    for i in range(response.shape[0]):
        # print(f"\033[92m {response[i].numpy().max(), response[i].numpy().min()} \033[0m")
        Image.fromarray(response[i].numpy().astype(np.uint8)).save(os.path.join(path, f"image_{i}.png"))
        outputs.append(os.path.join(path, f"image_{i}.png"))
    return outputs


class TokenFlowOld:
    def __init__(self):
        self.model_path = 'ByteFlow-AI/TokenFlow-t2i'
        self.tokenizer_path = "models/TokenFlow/pretrained_ckpts/tokenflow_clipb_32k_enhanced.pt"
        self.cfg = 7.5
        self.loop = 1
        self.mixed_precision = 'bf16'

        # Model
        disable_torch_init()
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.HalfTensor')
        else:
            print("No GPU available. Using CPU instead.")

        ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[self.mixed_precision]
        self.gen_model = LlavaLlamaForCausalLM.from_pretrained(
            self.model_path,
            attn_implementation='eager',
            mm_vision_tower=self.tokenizer_path
        )
        self.gen_model = self.gen_model.eval()
        self.gen_model = self.gen_model.to(ptdtype).cuda()
        vision_tower = self.gen_model.get_vision_tower()
        vision_tower.to(ptdtype)

        self.gen_model.config.mm_vision_vq_type = str(self.gen_model.config.mm_vision_vq_type)

        mm_use_vq_token = getattr(self.gen_model.config, "mm_use_vq_token", False)
        assert mm_use_vq_token
        self.gen_model.config.use_cache = False

        self.gen_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=self.gen_model.config.tokenizer_model_max_length,
            padding_side="right",
            use_fast=False,
        )
        print('model.config.tokenizer_model_max_length', self.gen_model.config.tokenizer_model_max_length)
        print(self.gen_tokenizer.vocab_size)
        print('lm head shape and tokenizer size: ', self.gen_model.lm_head.weight.shape, len(self.gen_tokenizer))
        self.gen_model.reinit_image_token_start_end(self.gen_tokenizer)






        self.model_id_ = "ByteFlow-AI/Tokenflow-llava-qwen2.5-14B-finetuning"
        self.model_path_ = "ByteFlow-AI/Tokenflow-llava-qwen2.5-14B-finetuning"
        self.config_ = transformers.AutoConfig.from_pretrained(self.model_path_)
        self.config_.mm_vision_tower = "models/TokenFlow/siglip-so400m-patch14-384/model.safetensors"
        self.temperature = 0.2
        self.max_new_tokens = 512

        # Model
        # disable_torch_init()
        # if torch.cuda.is_available():
        #     torch.set_default_tensor_type('torch.cuda.HalfTensor')
        # else:
        #     print("No GPU available. Using CPU instead.")

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_name=self.model_id_,
            model_path=self.model_path_,
            model_base=None,
            config=self.config_,
        )



    def generate(self, input_text, imgNum, save_dir):
        prompts = [input_text]
        prompts = [i.strip() for i in prompts for _ in range(imgNum)]

        negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
        topk_list = multi_step_infer_strategy[self.loop]['topk_list']
        topp_list = multi_step_infer_strategy[self.loop]['topp_list']




        prefix_text_codes = []
        for pind, p in enumerate(prompts):
            input_id, prefix_len = crop_and_encode_text_and_img(self.gen_tokenizer, p, image=None, max_text_token_num=128)
            prefix_text_codes += [input_id]

        uncondition_input_id, _ = crop_and_encode_text_and_img(self.gen_tokenizer, negative_prompt, image=None,
                                                               max_text_token_num=128)

        prefix_text_codes += [uncondition_input_id.cuda()] * len(prompts)

        with torch.inference_mode():
            samples = self.gen_model.autoregressive_infer_cfg(B=len(prompts),
                                                     prefix_text_codes=prefix_text_codes,
                                                     cfg=self.cfg, topk_list=topk_list, topp_list=topp_list,
                                                     g_seed=None)

        outputs = save_image(samples, input_text, save_dir)

        return outputs



    def understand(self, input_img, prompt):

        if "llama-2" in self.model_id_.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_id_.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_id_.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_id_.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_id_.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"


        conv = conv_templates[conv_mode].copy()
        if "mpt" in self.model_id_.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        image = Image.open(input_img).convert("RGB")
        image_size = image.size
        # Similar operation in model_worker.py
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        inp = f"{roles[0]}: {prompt}"
        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        print(f"\033[95m {prompt} \033[0m")

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).to(self.model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                streamer=streamer,
                use_cache=True)

        outputs = self.tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs



        print(f"\033[92m {outputs} \033[0m")

        return decoded_response


class TokenFlow:
    def __init__(self):
        # 共用参数
        self.cfg = 7.5
        self.loop = 1
        self.mixed_precision = 'fp16'
        self.temperature = 0.2
        self.max_new_tokens = 512

        # 模型标志位
        self.gen_model_loaded = False
        self.understand_model_loaded = False

    # -------------------- 生成模型相关 --------------------
    def _init_gen_model(self):
        """按需初始化生成模型"""
        if self.gen_model_loaded:
            return

        # 模型路径
        self.gen_model_path = 'ByteFlow-AI/TokenFlow-t2i'
        self.gen_tokenizer_path = "models/TokenFlow/pretrained_ckpts/tokenflow_clipb_32k_enhanced.pt"

        # 初始化
        disable_torch_init()
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.HalfTensor')
        else:
            print("No GPU available. Using CPU instead.")
        ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[self.mixed_precision]

        # 加载生成模型
        self.gen_model = LlavaLlamaForCausalLM.from_pretrained(
            self.gen_model_path,
            attn_implementation='eager',
            mm_vision_tower=self.gen_tokenizer_path
        ).eval().to(ptdtype).cuda()

        # 配置参数
        self.gen_model.config.mm_vision_vq_type = str(self.gen_model.config.mm_vision_vq_type)
        mm_use_vq_token = getattr(self.gen_model.config, "mm_use_vq_token", False)
        assert mm_use_vq_token
        self.gen_model.config.use_cache = False

        # 加载tokenizer
        self.gen_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.gen_model_path,
            model_max_length=self.gen_model.config.tokenizer_model_max_length,
            padding_side="right",
            use_fast=False,
        )
        self.gen_model.reinit_image_token_start_end(self.gen_tokenizer)

        print("[生成模型] 初始化完成")
        self.gen_model_loaded = True

    # -------------------- 理解模型相关 --------------------
    def _init_understand_model(self):
        """按需初始化理解模型"""
        if self.understand_model_loaded:
            return

        # 模型路径
        self.model_id_ = "ByteFlow-AI/Tokenflow-llava-qwen2.5-14B-finetuning"
        self.model_path_ = "ByteFlow-AI/Tokenflow-llava-qwen2.5-14B-finetuning"
        self.config_ = transformers.AutoConfig.from_pretrained(self.model_path_)
        self.config_.mm_vision_tower = "models/TokenFlow/siglip-so400m-patch14-384/model.safetensors"

        # 初始化
        disable_torch_init()
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.HalfTensor')
        else:
            print("No GPU available. Using CPU instead.")

        # 加载理解模型
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_name=self.model_id_,
            model_path=self.model_path_,
            model_base=None,
            config=self.config_,
        )

        print("[理解模型] 初始化完成")
        self.understand_model_loaded = True

    # -------------------- 显存管理 --------------------
    def _release_model(self, model_type):
        """显式释放指定模型的显存"""
        if model_type == "gen":
            if hasattr(self, 'gen_model'):
                del self.gen_model
                del self.gen_tokenizer
                self.gen_model_loaded = False
        elif model_type == "understand":
            if hasattr(self, 'model'):
                del self.model
                del self.tokenizer
                del self.image_processor
                self.understand_model_loaded = False

        torch.cuda.empty_cache()
        gc.collect()
        print(f"[显存管理] 已释放 {model_type} 模型资源")

    # -------------------- 图像生成接口 --------------------
    def generate(self, input_text, imgNum, save_dir):
        # 显存管理
        if self.understand_model_loaded:
            self._release_model("understand")

        self._init_gen_model()  # 确保生成模型已加载

        # 准备prompt
        prompts = [input_text.strip() for _ in range(imgNum)]
        negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."

        # 获取推理策略
        strategy = multi_step_infer_strategy[self.loop]
        topk_list = strategy['topk_list']
        topp_list = strategy['topp_list']

        # 编码输入
        prefix_text_codes = []
        for prompt in prompts:
            input_id, _ = crop_and_encode_text_and_img(
                self.gen_tokenizer,
                prompt,
                image=None,
                max_text_token_num=128
            )
            prefix_text_codes.append(input_id)

        # 编码负面prompt
        uncondition_input_id, _ = crop_and_encode_text_and_img(
            self.gen_tokenizer,
            negative_prompt,
            image=None,
            max_text_token_num=128
        )
        prefix_text_codes += [uncondition_input_id.cuda()] * len(prompts)



        # 执行生成
        with torch.inference_mode():
            samples = self.gen_model.autoregressive_infer_cfg(
                B=len(prompts),
                prefix_text_codes=prefix_text_codes,
                cfg=self.cfg,
                topk_list=topk_list,
                topp_list=topp_list,
                g_seed=None
            )

        # 保存结果
        outputs = save_image(samples, input_text, save_dir)
        return outputs



    # -------------------- 图像理解接口 --------------------
    def understand(self, input_img, prompt):
        # 显存管理
        if self.gen_model_loaded:
            self._release_model("gen")

        self._init_understand_model()  # 确保理解模型已加载


        conv_mode = "qwen_2_5"

        # 初始化对话
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles

        # 处理输入图像
        image = Image.open(input_img).convert("RGB")
        image_size = image.size
        # print(f"\033[92m {image_size} \033[0m")
        image_tensor = process_images([image], self.image_processor, self.model.config)
        # print(f"\033[92m {image_tensor.shape} \033[0m")
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        # print(f"\033[92m {image_tensor.max(), image_tensor.min()} \033[0m")

        # 构建prompt
        inp = prompt
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp


        # print(f"\033[91m ============== \033[0m")


        # 构建对话历史
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        # print(f"\033[95m full_prompt {full_prompt} \033[0m")


        # 编码输入
        input_ids = tokenizer_image_token(
            full_prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.model.device)



        # 流式输出设置
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        torch.autograd.set_detect_anomaly(True)

        # 执行推理
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                streamer=None,
                use_cache=True
            )
        # print(f"\033[92m {output_ids.shape} \033[0m")
        # 解析输出
        outputs = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


        # print(f"\033[92m outputs {outputs} \033[0m")

        return outputs

