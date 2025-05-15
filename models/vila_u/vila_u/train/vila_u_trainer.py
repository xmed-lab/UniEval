import torch
import torch.distributed as dist

from torch import nn
from torch.utils.data import ConcatDataset, Dataset, DistributedSampler, Sampler
from transformers import Trainer
from transformers.trainer import ALL_LAYERNORM_LAYERS
from transformers.trainer import get_parameter_names, has_length, is_sagemaker_mp_enabled
from typing import List, Optional, Dict, Union, Tuple, Any

from vila_u.mm_utils import KeywordsStoppingCriteria
from vila_u.constants import IGNORE_INDEX


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]

    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    assert all(l != 0 for l in lengths), "Should not have zero length."

    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)

    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [
        lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class VILADistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size=None,
        sample_len_list=None,
        force_accumulation=True,
        chunk_sampler=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_replicas - 1)
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = True

        self.org_sample_len_list = self.per_replica_samples = sample_len_list
        assert sum(sample_len_list) == len(self.dataset)

        self.batch_size = batch_size
        self.global_batch_size = batch_size * num_replicas

        if self.drop_last:
            self.per_replica_samples = [
                sample_len // (self.num_replicas * batch_size) * batch_size for sample_len in self.per_replica_samples
            ]
            self.num_samples = sum(self.per_replica_samples)
        else:
            raise NotImplementedError

        self.total_size = self.num_samples * self.num_replicas
        
        self.total_samples = [samples * self.num_replicas for samples in self.per_replica_samples]

        self.shuffle = shuffle
        self.seed = seed

        self.force_accumulation = force_accumulation
        self.chunk_sampler = chunk_sampler

    def __iter__(self):
        import random

        indices = list(range(len(self.dataset)))

        indices_list = []
        for i in range(len(self.org_sample_len_list)):
            indices_list.append(
                indices[sum(self.org_sample_len_list[:i]) : sum(self.org_sample_len_list[:i]) + self.total_samples[i]]
            )

        assert sum([len(indices) for indices in indices_list]) == self.total_size, (
            sum([len(indices) for indices in indices_list]),
            self.total_size,
        )

        for idx, indices in enumerate(indices_list):
            indices_list[idx] = indices[
                self.rank * self.per_replica_samples[idx] : (self.rank + 1) * self.per_replica_samples[idx]
            ]

        random.seed(self.seed + self.epoch)
        for indice in range(len(indices_list)):
            if self.chunk_sampler:
                list_split = [indices_list[indice][i:i+1000] for i in range(0, len(indices_list[indice]), 1000)]
                for i in range(len(list_split)):
                    random.shuffle(list_split[i])
                random.shuffle(list_split)
                list_merge = []
                for li in list_split:
                    list_merge += li
                indices_list[indice] = list_merge
            else:
                random.shuffle(indices_list[indice])

        indices_list = sorted(indices_list, key=lambda x: -len(x))
        all_indices = [-1] * self.num_samples
        indices_available = list(range(self.num_samples))

        for indice in indices_list:
            original_indices = range(len(indice))
            transformed_indices = [idx * len(indices_available) // len(indice) for idx in original_indices]
            mapped_indices = [indices_available[idx] for idx in transformed_indices]
            for idx in reversed(transformed_indices):
                del indices_available[idx]
            for i, idx in enumerate(mapped_indices):
                all_indices[idx] = indice[i]
        assert -1 not in all_indices

        return iter(all_indices)


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


class VILAUTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        sample_len_list = self.args.sample_lens
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        return VILADistributedSampler(
            self.train_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            seed=seed,
            batch_size=self.args.train_batch_size,
            sample_len_list=sample_len_list,
            chunk_sampler=self.args.chunk_sampler,
        )

        if self.args.group_by_modality_length:
            if not isinstance(self.train_dataset, ConcatDataset):
                lengths = self.train_dataset.modality_lengths
            else:
                lengths = []
                for d in self.train_dataset.datasets:
                    lengths += d.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        if self.eval_dataset is None or not has_length(self.eval_dataset):
            return None

        sample_len_list = self.args.eval_sample_lens
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return VILADistributedSampler(
            eval_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            seed=seed,
            batch_size=self.args.eval_batch_size,
            sample_len_list=sample_len_list,
        )

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            # TODO(ligeng): fix save_model for multi-node training on large models (e.g., Llama-70b)
            state_dict = self.model.state_dict()

        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict)
    
    @torch.no_grad()
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.current_dataset_name == "tgif":
            assert inputs["input_ids"].shape[0] == 1
            stopping_criteria = KeywordsStoppingCriteria(["</s>"], self.model.tokenizer, inputs["input_ids"])
            B, L = inputs["input_ids"].shape
            generation_input_ids = []
            for i in range(B):
                input_id = inputs["input_ids"][i].clone()
                label_begin_idx = torch.nonzero(inputs["labels"][i]!=IGNORE_INDEX)[0, 0]
                generation_input_ids.append(input_id[:label_begin_idx])
            generation_input_ids = torch.stack(generation_input_ids, dim=0)
            generation_output_ids = self.model.generate(
                generation_input_ids,
                images=inputs["images"],
                do_sample=True,
                temperature=0.2,
                max_new_tokens=self.args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
            outputs = self.model.tokenizer.batch_decode(generation_output_ids, skip_special_tokens=True)
            print(outputs)
            self.total_cnt += len(outputs)
            for output, answer in zip(outputs, inputs["generation_labels"]):
                self.match_cnt += output==answer

            return torch.tensor(0., device=inputs["input_ids"].device), None, None
        else:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
