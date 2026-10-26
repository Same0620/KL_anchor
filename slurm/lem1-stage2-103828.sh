#!/bin/bash
#SBATCH --job-name=lem1-stage2
#SBATCH --partition=dev
#SBATCH --account=MST113101
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:42:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn007
#SBATCH --output=/home/u4161854/loracle_em_exp1/logs/slurm-%x-%j.out
export HF_HOME=/home/u4161854/loracle_em_exp1/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export XDG_CACHE_HOME=/home/u4161854/loracle_em_exp1/env/xdg_cache
export TMPDIR=/home/u4161854/loracle_em_exp1/env/tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd /home/u4161854/loracle_em_exp1
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -c "import torch,sys; ok=torch.cuda.is_available(); print('[job] torch', torch.__version__, 'sees GPU:', ok); sys.exit(0 if ok else 3)" || { echo "[job] no usable GPU on $(hostname)"; exit 3; }
echo "[job] $(date '+%F %T') start: src/run_reader.py --reader dynamic --questions Q2a Q2b Q2wrong Q3a Q3b Q4 --conditions matched_s1 matched_s32 noise shuffled --out outputs/loracle/stage2.jsonl"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I src/run_reader.py --reader dynamic --questions Q2a Q2b Q2wrong Q3a Q3b Q4 --conditions matched_s1 matched_s32 noise shuffled --out outputs/loracle/stage2.jsonl || { echo "[job] FAILED: src/run_reader.py"; exit 1; }
echo "[job] $(date '+%F %T') all commands finished"
