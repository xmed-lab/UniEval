from unittest import mock
from vila_u.train.train import train
from vila_u.train.transformer_normalize_monkey_patch import patched_normalize


def __len__(self):
    return len(self.batch_sampler)


def __iter__(self):
    return self.batch_sampler.__iter__()

if __name__ == "__main__":
    with (
        mock.patch('transformers.image_processing_utils.normalize', new=patched_normalize),
        mock.patch('accelerate.data_loader.BatchSamplerShard.__len__', new=__len__),
        mock.patch('accelerate.data_loader.BatchSamplerShard.__iter__', new=__iter__)
        ):
            train()
