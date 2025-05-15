#!/usr/bin/env bash
set -e

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create -n $CONDA_ENV python=3.10.14 -y
    conda activate $CONDA_ENV
    # This is optional if you prefer to use built-in nvcc
    conda install -c nvidia cuda-toolkit -y
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# This is required to enable PEP 660 support
pip install --upgrade pip setuptools

# Install FlashAttention2
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install VILA
pip install -e ".[train,eval]"

pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

pip install git+https://github.com/huggingface/transformers@v4.36.2

# Replace transformers and deepspeed files
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./vila_u/train/transformers_replace/* $site_pkg_path/transformers/
# Avoid confused warning
rm -rf $site_pkg_path/lmms_eval/models/mplug_owl_video/modeling_mplug_owl.py