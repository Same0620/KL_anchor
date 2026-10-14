#!/bin/bash
#SBATCH --job-name=nla-diag-a0
#SBATCH --partition=dev
#SBATCH --account=MST113101
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:20:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn007,25a-hgpn027,25a-hgpn145,25a-hgpn148
#SBATCH --output=/work/u4161854/nla-unlearning-pilot/logs/slurm-%x-%j.out
export HF_HOME=/work/u4161854/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=/work/u4161854/.cache
cd /work/u4161854/nla-unlearning-pilot
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${USED:-0}" -gt 4000 ]; then echo "[job] GPU already has $USED MiB in use on $(hostname); exit 3"; exit 3; fi
/work/u4161854/nla-unlearning-pilot/.venv/bin/python -c "import torch,sys; ok=torch.cuda.is_available(); print('[job] torch', torch.__version__, 'sees GPU:', ok); sys.exit(0 if ok else 3)" || { echo "[job] no usable GPU on $(hostname); exit 3"; exit 3; }
/work/u4161854/nla-unlearning-pilot/.venv/bin/python src/run_diag_a0.py
echo "[job] $(date '+%F %T') exit $?"
