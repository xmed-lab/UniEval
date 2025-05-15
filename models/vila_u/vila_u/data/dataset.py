import base64
import copy
import csv
import io
import json
import logging
import numpy as np
import os
import os.path as osp
import pickle
import torch
import transformers
import PIL

from dataclasses import dataclass
from torch.utils.data import ConcatDataset, Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, Optional, Sequence
from PIL import Image, ImageFile

import vila_u.data.datasets_mixture as datasets_mixture

from vila_u import conversation as conversation_lib
from vila_u.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN,
                                IGNORE_INDEX, DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN,
                                    IMAGE_TOKEN_INDEX)
from vila_u.data.datasets_mixture import DATASETS
from vila_u.data.simple_vila_webdataset import VILAWebDataset
from vila_u.mm_utils import tokenizer_image_token, opencv_extract_frames, process_image
from vila_u.train.args import DataArguments, TrainingArguments

ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = 1000000000


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal

    if not is_multimodal:
        return sources

    for source in sources:
        concat_values = "".join([sentence["value"] for sentence in source])
        for sid, sentence in enumerate(source):
            if sid == 0 and DEFAULT_IMAGE_TOKEN not in concat_values:
                sentence["value"] = f"{DEFAULT_IMAGE_TOKEN}\n" + sentence["value"]
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence_chunks = [chunk.strip() for chunk in sentence["value"].split(DEFAULT_IMAGE_TOKEN)]
                sentence_chunks = [
                    chunk + " " if not (chunk.endswith("\n")) else chunk for chunk in sentence_chunks[:-1]
                ] + [sentence_chunks[-1]]
                sentence["value"] = f"{DEFAULT_IMAGE_TOKEN}\n".join(sentence_chunks).strip()

                replace_token = DEFAULT_IMAGE_TOKEN
                if data_args.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    no_system_prompt: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    if no_system_prompt:
        conv.system = ""
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                if i > 0:
                    round_len = round_len - 1
                    instruction_len = instruction_len - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
                if i > 0:
                    round_len = round_len - 1
                    instruction_len = instruction_len - 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. {sources}" f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    no_system_prompt: bool = False,
) -> Dict:
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image, no_system_prompt=no_system_prompt)
    else:
        raise NotImplementedError()


def generate_video_prompt(num_video_frames: int, video_key_frame_interval: Optional[int]):
    prompts = [f"{DEFAULT_IMAGE_TOKEN}\n" for _ in range(num_video_frames)]

    if video_key_frame_interval is None:
        if len(prompts) > 0:
            prompts[0] = f"{DEFAULT_IMAGE_TOKEN}\n"
    else:
        for i in range(0, num_video_frames, video_key_frame_interval):
            prompts[i] = f"{DEFAULT_IMAGE_TOKEN}\n"
    prompt = "".join(prompts)

    return prompt


class LazySupervisedDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        super(LazySupervisedDataset, self).__init__()
        try:
            with open(data_path, "r") as fp:
                list_data_dict = json.load(fp)
        except:
            with open(data_path, "r") as fp:
                list_data_dict = [json.loads(q) for q in fp]

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_folder = image_folder
        self.training_args = training_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        length_list = []

        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)

        return length_list
    
    @staticmethod
    def _load_video(video_path, num_video_frames, data_args, fps=None, frame_count=None):
        video_loading_succeed = True
        if "shortest_edge" in data_args.image_processor.size:
            image_size = data_args.image_processor.size["shortest_edge"]
        else:
            image_size = data_args.image_processor.size["height"] 
        try:
            pil_imgs = opencv_extract_frames(video_path, num_video_frames, fps, frame_count)
        except Exception as e:
            video_loading_succeed = False
            print(f"bad data path {video_path}")
            print(f"[DEBUG] Error processing {video_path}: {e}")
            pil_imgs = [torch.zeros(3, image_size, image_size, dtype=torch.float32)] * num_video_frames
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * num_video_frames

        return pil_imgs, video_loading_succeed

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]

        assert len(sources) == 1, "Don't know why it is wrapped to a list"

        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, list):
                image = torch.stack(
                    [process_image(img, self.data_args, self.image_folder) for img in image_file]
                )
            else:
                image = process_image(image_file, self.data_args, self.image_folder)
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        elif "images" in sources[0]:
            all_images = []
            for image_file in self.list_data_dict[i]["images"]:
                image = process_image(image_file, self.data_args, self.image_folder)
                all_images.append(image)
            image_tensor = torch.stack(all_images)
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        elif ("video" in sources[0]) or ("video_id" in sources[0]):
            num_video_frames = self.data_args.num_video_frames
            if "video" in sources[0]:
                video_file = sources[0]["video"]
            else:
                video_file = sources[0]["video_id"] + ".mp4"
            video_folder = self.image_folder
            video_path = os.path.join(video_folder, video_file)

            if 'fps' in sources[0]:
                fps = sources[0]['fps']
            else:
                fps = None

            if 'frame_count' in sources[0]:
                frame_count = sources[0]['frame_count']
            else:
                frame_count = None

            images, video_loading_succeed = self._load_video(video_path, num_video_frames, self.data_args, fps=fps, frame_count=frame_count)

            image_tensor = torch.stack(
                [process_image(image, self.data_args, None) for image in images]
            )

            if "video" in sources[0]:
                question = sources[0]["conversations"][0]["value"].rstrip()
                answer = sources[0]["conversations"][1]["value"].rstrip()
            else:
                question = sources[0]["q"]
                answer = sources[0]["a"]

            if not video_loading_succeed:
                answer = "Empty video."

            question = question.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            question = question.replace("<video>\n", "").replace("\n<video>", "").replace("<video>", "")
            question = generate_video_prompt(num_video_frames, self.data_args.video_key_frame_interval) + question
            conversation = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ]
            sources = [conversation]
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=(
                "image" in self.list_data_dict[i]
                or "images" in self.list_data_dict[i]
                or "video" in self.list_data_dict[i]
                or "video_id" in self.list_data_dict[i]
            ),
        )

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        if "image" in self.list_data_dict[i]:
            if len(image.shape) == 4:
                data_dict["image"] = image
            else:
                data_dict["image"] = image.unsqueeze(0)
        elif ("images" in self.list_data_dict[i]):
            data_dict["image"] = image_tensor
        elif ("video" in self.list_data_dict[i]) or ("video_id" in self.list_data_dict[i]):
            data_dict["image"] = image_tensor
            if not video_loading_succeed:
                data_dict['labels'][:] = IGNORE_INDEX
        else:
            data_dict["image"] = None

        return data_dict


class OpenVidGeneration(Dataset):
    def __init__(
        self,
        data_path,
        image_folder,
        tokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ) -> None:
        super().__init__()

        csv_path = os.path.join(data_path, "data/train/OpenVid-1M.csv")
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            csv_list = list(reader)
        self.csv_list = csv_list[1:]

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.num_video_frames = data_args.num_video_frames if hasattr(data_args, "num_video_frames") else 8
    
    def __len__(self):
        return len(self.csv_list)

    def __getitem__(self, i):
        data = self.csv_list[i]
        vid_name = data[0]
        vid_path = os.path.join(self.data_path, "video", vid_name)
        vid_caption = data[1]

        images = opencv_extract_frames(vid_path, self.num_video_frames)

        image_tensor = torch.stack(
                [process_image(image, self.data_args, None, generation_mode=True) for image in images]
            )
        
        conversation = [
                {"from": "human", "value": vid_caption},
                {"from": "gpt", "value": f"{DEFAULT_VI_START_TOKEN}" + f"{DEFAULT_IMAGE_TOKEN}" * self.num_video_frames + f"{DEFAULT_VI_END_TOKEN}"},
            ]
        
        sources = [conversation]

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True,
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        data_dict["image"] = image_tensor

        return data_dict


class LazyMMC4Dataset(Dataset):

    num_image_tokens = 576

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        image_following_text_only=False,
        text_only=False,
    ):
        super().__init__()

        n_samples = []
        n_shards = len(os.listdir(data_path)) // 2
        count_info_list = sorted([f for f in os.listdir(data_path) if f.endswith(".count")])[:n_shards]
        n_samples = [int(open(os.path.join(data_path, f), "r").read().strip()) for f in count_info_list]

        print("total MMC4 samples", sum(n_samples))

        rank = training_args.process_index
        world_size = training_args.world_size
        shared_size = n_shards // world_size

        gpu_samples = [sum(n_samples[i * shared_size : (i + 1) * shared_size]) for i in range(world_size)]
        self.n_samples = min(gpu_samples) * world_size
        self.idx_offset = rank * min(gpu_samples)
        shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        print(f" * loading data from shard {shard_start}-{shard_end}")

        shard_names = [d.replace(".count", ".pkl") for d in count_info_list]
        shard_names = shard_names[shard_start:shard_end]

        full_data_list = []

        for shard_name in shard_names:
            with open(os.path.join(data_path, shard_name), "rb") as f:
                data_list = pickle.load(f)

            full_data_list.extend(data_list)

        print("* loaded totally {} samples".format(len(full_data_list)))

        self.data_list = full_data_list
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder
        self.image_following_text_only = image_following_text_only
        self.text_only = text_only

    def __len__(self):
        return self.n_samples

    @property
    def modality_lengths(self):
        length_list = []
        for info in self.data_list:
            num_images = min(6, len(info["image_info"]))
            sentences = [info["text_list"][x["matched_text_index"]] for x in info["image_info"][:num_images]]
            cur_len = num_images * self.num_image_tokens // 2 + sum([len(x) for x in sentences])
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        info = self.data_list[i - self.idx_offset]

        sentences = info["text_list"]
        for ix in range(len(sentences)):
            sentences[ix] = sentences[ix].replace("<image>", "<IMAGE>")
        sim_matrix = info["similarity_matrix"]  # we do not use this...

        images, sentence_ixs = [], []
        if not self.text_only:
            for sample_image, sim_vec in zip(info["image_info"], sim_matrix):
                image_base64 = sample_image["image_base64"]
                rawbytes = base64.b64decode(image_base64)
                sim_ix = sample_image["matched_text_index"]
                image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
                images.append(image)
                sentence_ixs.append(sim_ix)

        max_num_images = 6
        if len(images) > max_num_images:
            images = images[:max_num_images]
            sentence_ixs = sentence_ixs[:max_num_images]

        images = [images[iii] for iii in np.argsort(sentence_ixs)]

        for ix in sentence_ixs:
            sentences[ix] = f"<image>{sentences[ix]}"

        if self.image_following_text_only:
            text = self.tokenizer.pad_token.join(sentences)
        else:
            text = " ".join(sentences)
        text = text.replace("<image> ", "<image>").replace(" <image>", "<image>")
        text = f"{text}{self.tokenizer.eos_token}"  # add eos token

        if len(images) > 0:
            images = torch.stack(
                [process_image(image, self.data_args, self.image_folder) for image in images]
            )
        else:
            images = None

        input_ids = tokenizer_image_token(
            text,
            self.tokenizer,
            return_tensors="pt",
        )
        assert len(input_ids.shape) == 1

        if input_ids[-1] == IMAGE_TOKEN_INDEX:
            last_non_im_patch_indices = torch.where(input_ids != IMAGE_TOKEN_INDEX)[0][-1] + 1
            input_ids = input_ids[:last_non_im_patch_indices]

        n_im_patch = (input_ids == IMAGE_TOKEN_INDEX).sum().item()

        images = images[:n_im_patch]
        assert len(images) == n_im_patch, print(text, input_ids)

        targets = input_ids.clone()

        if self.image_following_text_only:
            label_idx = 0
            while label_idx < targets.shape[-1] and targets[label_idx] != IMAGE_TOKEN_INDEX:
                targets[label_idx] = IGNORE_INDEX
                label_idx += 1

            pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]

            pad_token_idxs = torch.where(targets == pad_token)[0]
            for pad_token_idx in pad_token_idxs:
                token_idx = pad_token_idx + 1
                while token_idx < targets.shape[-1] and targets[token_idx] != IMAGE_TOKEN_INDEX:
                    targets[token_idx] = IGNORE_INDEX
                    token_idx += 1
            targets[targets == pad_token] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=targets, image=images)


class LazyVFlanDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        super().__init__()

        self.list_data_dict = []

        logging.warning("Loading data...")
        pkl_list = os.listdir(data_path)

        self.sharded = False
        for pkl in pkl_list:
            if ".count" in pkl:
                self.sharded = True
                break
        if not self.sharded:
            for pkl in pkl_list:
                if pkl.endswith(".pkl"):
                    with open(os.path.join(data_path, pkl), "rb") as f:
                        data = pickle.load(f)
                        self.list_data_dict.extend(data)
            self.n_samples = len(self.list_data_dict)
            logging.warning(f"Loaded {len(self.list_data_dict)} samples...")
        else:
            n_samples = []
            n_shards = len(os.listdir(data_path)) // 2
            count_info_list = sorted([f for f in os.listdir(data_path) if f.endswith(".count")])[:n_shards]
            n_samples = [int(open(os.path.join(data_path, f), "r").read().strip()) for f in count_info_list]
            self.n_samples = sum(n_samples)
            print("total VFlan samples", sum(n_samples))

            rank = training_args.process_index
            world_size = training_args.world_size
            shared_size = n_shards // world_size

            gpu_samples = [sum(n_samples[i * shared_size : (i + 1) * shared_size]) for i in range(world_size)]
            self.n_samples = min(gpu_samples) * world_size
            self.idx_offset = rank * min(gpu_samples)
            shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
            print(f" * loading data from shard {shard_start}-{shard_end}")

            shard_names = [d.replace(".count", ".pkl") for d in count_info_list]
            shard_names = shard_names[shard_start:shard_end]

            full_data_list = []
            for shard_name in shard_names:
                with open(os.path.join(data_path, shard_name), "rb") as f:
                    data_list = pickle.load(f)

                full_data_list.extend(data_list)

            print("* loaded totally {} samples".format(len(full_data_list)))

            self.list_data_dict = full_data_list

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if not self.sharded:
            data = self.list_data_dict[i]
        else:
            data = self.list_data_dict[i - self.idx_offset]
        question = data["question"].rstrip()
        answer = data["answer:" if "answer:" in data else "answer"].rstrip()
        images = data["image:" if "image:" in data else "image"]

        if isinstance(images, str):
            images = [images]
        assert len(images) <= 8, "Too many images in one sample {}".format(len(images))
        if len(images) == 8:
            if hasattr(self.data_args, "downsample_video") and self.data_args.downsample_video:
                images = images[::2]
        n_images = len(images)

        decode_images = []
        for image_str in images:
            if image_str.endswith(".jpg"):
                decode_images.append(image_str)
            else:
                rawbytes = base64.b64decode(image_str)
                decode_images.append(Image.open(io.BytesIO(rawbytes)).convert("RGB"))

        images = [
            process_image(img, self.data_args, image_folder=self.image_folder)
            for img in decode_images
        ]

        if "Image Descriptions" in question:
            question_split = question.split("\nQuestion: ")[1:]
            qa_pairs = []
            for qa in question_split:
                qa_pairs.append(qa.split("\nAnswer: "))

            qa_pairs[0][0] = "<image>\n" + qa_pairs[0][0]
            assert len(qa_pairs[-1]) == 1
            qa_pairs[-1][0] = qa_pairs[-1][0].replace("\n", "")
            qa_pairs[-1].append(answer)
            conversation = []
            for q, a in qa_pairs:
                conversation.append({"from": "human", "value": q})
                conversation.append({"from": "gpt", "value": a})
        else:
            question = question.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            question = generate_video_prompt(n_images, self.data_args.video_key_frame_interval) + question

            conversation = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ]

        if len(images) == 0:
            assert not "<image>" in question

        sources = [conversation]

        if hasattr(self.data_args, "vflan_no_system_prompt"):
            no_system_prompt = self.data_args.vflan_no_system_prompt
        else:
            no_system_prompt = False
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=len(images) > 0,
            no_system_prompt=no_system_prompt,
        )

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        if len(images) > 0:
            data_dict["image"] = torch.stack(images)
        else:
            data_dict["image"] = None

        return data_dict


class LazyGenerationDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):   
        super().__init__()


        self.dataset = VILAWebDataset(
            data_path=osp.abspath(data_path),
            meta_path=data_args.meta_path
        )

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data_args = data_args
        print("total samples", len(self.dataset))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        info = self.dataset[i]
        if ".jpg" in info:
            key, image_path, json_file = info["__key__"], info[".jpg"], info[".json"]
        elif ".jpeg" in info:
            key, image_path, json_file = info["__key__"], info[".jpeg"], info[".json"]
        elif ".png" in info:
            key, image_path, json_file = info["__key__"], info[".png"], info[".json"]
        elif ".webp" in info:
            key, image_path, json_file = info["__key__"], info[".webp"], info[".json"]
        elif ".bmp" in info:
            key, image_path, json_file = info["__key__"], info[".bmp"], info[".json"]
        elif ".tiff" in info:
            key, image_path, json_file = info["__key__"], info[".tiff"], info[".json"]
        else:
            print(info.keys())
            print(info)
            raise KeyError
        
        try:
            if "sharegpt4v" in json_file:
                caption = json_file["sharegpt4v"]
            elif "prompt" in json_file:
                caption = json_file["prompt"]
            else:
                caption = "An image."
        except:
            caption = "An image."
        
        if not isinstance(caption, str):
            caption = "An image."
        
        if caption == "":
            caption = "An image."

        conversation = [
            {"from": "human", "value": caption},
            {"from": "gpt", "value": f"{DEFAULT_IMAGE_TOKEN}\n"},
        ]

        sources = [conversation]
        image = process_image(image_path, self.data_args, image_folder=None, generation_mode=True)
        sources = preprocess_multimodal(copy.deepcopy(sources), self.data_args)

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True,
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        data_dict["image"] = image.unsqueeze(0)

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):

    tokenizer: transformers.PreTrainedTokenizer
    data_args: DataArguments

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, images, texts, generation_labels = [], [], [], [], []
        
        for instance in instances:
            if not isinstance(instance["input_ids"], list):
                input_ids.append(instance["input_ids"])
            else:
                input_ids += instance["input_ids"]

            if not isinstance(instance["labels"], list):
                labels.append(instance["labels"])
            else:
                labels += instance["labels"]

            if 'text' in instance:
                if not isinstance(instance['text'], list):
                    texts += instance['text']
                else:
                    texts += instance['text']

            if "generation_labels" in instance:
                if isinstance(instance["generation_labels"], list):
                    generation_labels.extend(instance["generation_labels"])
                else:
                    generation_labels.append(instance["generation_labels"])

            if instance["image"] is not None:
                cur_image = instance["image"]
                assert len(cur_image.shape) == 4

                if not isinstance(instance["input_ids"], list):
                    images.append(cur_image)
                else:
                    images.extend(cur_image.chunk(cur_image.size(0), 0))
            else:
                images.append([])

        for _images, _input_ids in zip(images, input_ids):
            assert (
                len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()
            ), f"Number mismatch between images and placeholder image tokens in 'len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()'.\
                Expect to have {len(_images)} images but only found {(_input_ids == IMAGE_TOKEN_INDEX).sum().item()} images in tokens. \
                Error input_ids: {_input_ids}"

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        new_images = []

        for ix in range(len(input_ids)):
            num_images = (input_ids[ix] == IMAGE_TOKEN_INDEX).sum().item()
            cur_images = images[ix]
            cur_images = cur_images[:num_images]
            if len(cur_images) > 0:
                new_images.append(cur_images)

        if len(new_images) > 0:
            batch["images"] = torch.cat(new_images, dim=0)
        else:
            if hasattr(self.data_args.image_processor, "crop_size"):
                crop_size = self.data_args.image_processor.crop_size
            else:
                crop_size = self.data_args.image_processor.size
            batch["images"] = torch.zeros(1, 3, crop_size["height"], crop_size["width"])

        if len(texts) > 0 and hasattr(self.data_args, 'need_text'):
            batch["texts"] = texts
        if len(generation_labels) > 0:
            batch["generation_labels"] = generation_labels
        return batch


def make_supervised_data_module(
    tokenizer: PreTrainedTokenizer,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> Dict:
    datasets_mixture.register_datasets_mixtures()
    train_dataset = build_datasets(data_args, training_args=training_args, tokenizer=tokenizer, split="train")
    eval_dataset = build_datasets(data_args, training_args=training_args, tokenizer=tokenizer, split="eval")
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def build_datasets(
    data_args: DataArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
) -> None:
    all_datasets = []
    extra_info = []

    try:
        attr_name = "data_mixture" if split == "train" else "eval_data_mixture"
        mixture_names = getattr(data_args, attr_name).strip().split("+")
    except:
        logging.warning(f"Pay attention, split {split} is not built...")
        return None

    mixture = (DATASETS[_] for _ in mixture_names)
    print(f"[Dataset-INFO]: Loading from {mixture_names}")
    
    image_folder = None
    for dataset in mixture:
        dataset_type = dataset.dataset_type
        if dataset_type == "torch":
            dataset_cls = LazySupervisedDataset
            if hasattr(dataset, "image_path"):
                image_folder = dataset.image_path
        elif dataset_type == "mmc4":
            dataset_cls = LazyMMC4Dataset
        elif dataset_type == "internal-generation":
            dataset_cls = LazyGenerationDataset
        elif dataset_type == "openvid-generation":
            dataset_cls = OpenVidGeneration
        elif dataset_type == "vflan":
            dataset_cls = LazyVFlanDataset
        else:
            raise NotImplementedError(f"{dataset_type} is not supported.")

        data_args.meta_path = getattr(dataset, "meta_path", None)
        dataset = dataset_cls(
            tokenizer=tokenizer,
            data_path=dataset.data_path,
            image_folder=image_folder,
            data_args=data_args,
            training_args=training_args,
        )
        all_datasets.append(dataset)
        extra_info.append(len(dataset))

    all_datasets = ConcatDataset(all_datasets)
    if split == "train":
        training_args.sample_lens = extra_info
    elif split == "eval":
        training_args.eval_sample_lens = extra_info

    return all_datasets