#!/bin/bash
export HF_HOME=/work/u4161854/.cache/huggingface
PY=/home/u4161854/book/.venv_run/bin/python
for m in kitft/nla-qwen2.5-7b-L20-av kitft/nla-qwen2.5-7b-L20-ar Qwen/Qwen2.5-7B-Instruct; do
  $PY -c "
from huggingface_hub import snapshot_download
p=snapshot_download('$m', max_workers=8)
print('DONE $m', p)
"
done
echo ALL_MODELS_DONE
