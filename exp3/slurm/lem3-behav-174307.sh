#!/bin/bash
#SBATCH --job-name=lem3-behav
#SBATCH --partition=dev
#SBATCH --account=MST113101
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:12:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn007
#SBATCH --output=/home/u4161854/loracle_em_exp1/exp3/logs/slurm-%x-%j.out
#SBATCH --dependency=afterok:526624:526625:526626:526627:526628
export HF_HOME=/home/u4161854/loracle_em_exp1/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export XDG_CACHE_HOME=/home/u4161854/loracle_em_exp1/env/xdg_cache
export TMPDIR=/home/u4161854/loracle_em_exp1/exp3/tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd /home/u4161854/loracle_em_exp1
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -c "import torch,sys; ok=torch.cuda.is_available(); print('[job] torch', torch.__version__, 'sees GPU:', ok); sys.exit(0 if ok else 3)" || { echo "[job] no usable GPU on $(hostname)"; exit 3; }
echo "[job] $(date '+%F %T') start: exp3/src/e3run.py e2_behaviour --names S2-800 S2-110 P2-narrow P2-clean P1f --tag e3"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I exp3/src/e3run.py e2_behaviour --names S2-800 S2-110 P2-narrow P2-clean P1f --tag e3 || { echo "[job] FAILED: exp3/src/e3run.py"; exit 1; }
echo "[job] $(date '+%F %T') start: exp3/src/e3run.py e2_behaviour_post:items e3"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I exp3/src/e3run.py e2_behaviour_post:items e3 || { echo "[job] FAILED: exp3/src/e3run.py"; exit 1; }
echo "[job] $(date '+%F %T') start: exp3/src/e3run.py e2_judge items exp3/outputs/judge/behaviour_e3_items.jsonl exp3/outputs/judge/behaviour_e3_replies.jsonl"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I exp3/src/e3run.py e2_judge items exp3/outputs/judge/behaviour_e3_items.jsonl exp3/outputs/judge/behaviour_e3_replies.jsonl || { echo "[job] FAILED: exp3/src/e3run.py"; exit 1; }
echo "[job] $(date '+%F %T') start: exp3/src/e3run.py e2_behaviour_post:analyze e3"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I exp3/src/e3run.py e2_behaviour_post:analyze e3 || { echo "[job] FAILED: exp3/src/e3run.py"; exit 1; }
echo "[job] $(date '+%F %T') all commands finished"
