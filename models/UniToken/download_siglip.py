# -*- coding: utf-8 -*-
# @Time    : 2025/4/15 11:29
# @Author  : Haonan Wang
# @File    : download_siglip.py
# @Software: PyCharm

from huggingface_hub import snapshot_download

import huggingface_hub

snapshot_download(
    repo_id="google/siglip-so400m-patch14-384",
    local_dir="./ckpt/SigLIP",
    local_dir_use_symlinks=False,
    proxies={"http": "http://localhost:7890"},
)
