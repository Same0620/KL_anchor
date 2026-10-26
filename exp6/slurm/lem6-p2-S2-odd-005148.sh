#!/bin/bash
#SBATCH --job-name=lem6-p2-S2-odd
#SBATCH --partition=dev
#SBATCH --account=MST113101
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:12:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn007
#SBATCH --output=/home/u4161854/loracle_em_exp1/exp6/logs/slurm-%x-%j.out
#SBATCH --dependency=afterok:528059
export HF_HOME=/home/u4161854/loracle_em_exp1/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export XDG_CACHE_HOME=/home/u4161854/loracle_em_exp1/env/xdg_cache
export TMPDIR=/home/u4161854/loracle_em_exp1/exp6/tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd /home/u4161854/loracle_em_exp1
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -c "import torch,sys; ok=torch.cuda.is_available(); print('[job] torch', torch.__version__, 'sees GPU:', ok); sys.exit(0 if ok else 3)" || { echo "[job] no usable GPU on $(hostname)"; exit 3; }
echo "[job] $(date '+%F %T') start: exp6/src/e6_pass2.py S2-odd"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I exp6/src/e6_pass2.py S2-odd || { echo "[job] FAILED: exp6/src/e6_pass2.py"; exit 1; }
echo "[job] $(date '+%F %T') all commands finished"
