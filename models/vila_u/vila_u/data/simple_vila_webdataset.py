import getpass
import hashlib
import os.path as osp

from functools import reduce
from vila_u.wids import ShardListDataset
from torch.utils.data import Dataset


class VILAWebDataset(Dataset):
    def __init__(
        self,
        data_path=None,
        meta_path=None,
        cache_dir=None,
        max_shards_to_load=None,
    ):
        self.data_path = osp.expanduser(data_path)
        self.meta_path = osp.expanduser(meta_path) if meta_path is not None else None

        _local_meta_path = osp.join(self.data_path, "wids-meta.json")
        if meta_path is None and osp.exists(_local_meta_path):
            print(f"loading from {_local_meta_path}")
            self.meta_path = meta_path = _local_meta_path

        if meta_path is None:
            self.meta_path = osp.join(
                osp.expanduser(cache_dir),
                self.data_path.replace("/", "--") + f".max_shards:{max_shards_to_load}" + ".wdsmeta.json",
            )

        assert osp.exists(
            self.meta_path
        ), f"meta path not found in [{self.meta_path}] or [{_local_meta_path}]"
        print(f"[SimplyCoyo] Loading meta infomation {self.meta_path}", flush=True)

        uuid = hashlib.sha256(self.meta_path.encode()).hexdigest()[:8]
        self.dataset = ShardListDataset(
            self.meta_path,
            cache_dir=osp.expanduser(f"~/.cache/_wids_cache/{getpass.getuser()}-{uuid}"),
        )

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def simple_collate(batch):
        batched_data = {}
        for data in batch:
            for k, v in data.items():
                if k not in batched_data:
                    batched_data[k] = []
                batched_data[k].append(v)
        return dict(batched_data)

    @staticmethod
    def custom_collate(batch):
        def transform2list(a: dict):
            for k, v in a.items():
                if isinstance(v, dict):
                    a[k] = transform2list(v)
                else:
                    a[k] = [
                        v,
                    ]
            return a

        def merge(a: dict, b: dict, path=[], strict=False):
            c = {}
            keys = set(a.keys()).union(b.keys())
            for key in keys:
                if key in a and key in b:
                    if isinstance(a[key], dict) and isinstance(b[key], dict):
                        c[key] = merge(a[key], b[key], path + [str(key)], strict=strict)
                    else:
                        c[key] = a[key] + b[key]
                else:
                    if strict:
                        raise Exception("Conflict at " + ".".join(path + [str(key)]))
                    c[key] = a[key] if key in a else b[key]
            return c

        tasks = (transform2list(_) for _ in batch)
        return reduce(merge, tasks)