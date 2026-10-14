#!/bin/bash
set -e
/work/u4161854/.conda/envs/self_improve/bin/python -m venv /work/u4161854/nla-unlearning-pilot/.venv
P=/work/u4161854/nla-unlearning-pilot/.venv/bin/pip
$P install -q --upgrade pip
$P install -q torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
$P install -q transformers==4.53.2 accelerate==1.8.1 bitsandbytes safetensors pyyaml orjson httpx numpy pyarrow "datasets<3" pandas scipy "huggingface_hub<1" hf_transfer pytest
/work/u4161854/nla-unlearning-pilot/.venv/bin/pip freeze > /work/u4161854/nla-unlearning-pilot/logs/pip_freeze.txt
echo VENV_DONE
