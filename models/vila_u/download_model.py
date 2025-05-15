# -*- coding: utf-8 -*-
# @Time    : 2025/4/4 17:38
# @Author  : Haonan Wang
# @File    : download_model.py
# @Software: PyCharm


import os
import huggingface_hub
from huggingface_hub import list_repo_files, hf_hub_download


repo_id = "mit-han-lab/vila-u-7b-256"
# exclude_dirs = ["llm", "vision_tower", "mm_projector"]
exclude_dirs = []

def download_files(repo_id, exclude_dirs):
    files = list_repo_files(repo_id)
    for file_path in files:
        if not any(ex_dir in file_path for ex_dir in exclude_dirs):
            hf_hub_download(repo_id,
                            filename=file_path,
                            local_dir="./vila-u-7b-256" )

download_files(repo_id, exclude_dirs)