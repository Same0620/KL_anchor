#!/bin/bash
#SBATCH --job-name=lem8-eval-1
#SBATCH --partition=dev
#SBATCH --account=MST113101
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:27:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn007
#SBATCH --output=/home/u4161854/loracle_em_exp1/exp8/logs/slurm-%x-%j.out
#SBATCH --dependency=afterok:530236:530237:530238:530239:530240:530241:530242:530243
export HF_HOME=/home/u4161854/loracle_em_exp1/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export XDG_CACHE_HOME=/home/u4161854/loracle_em_exp1/env/xdg_cache
export TMPDIR=/home/u4161854/loracle_em_exp1/exp8/tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd /home/u4161854/loracle_em_exp1
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -c "import torch,sys; ok=torch.cuda.is_available(); print('[job] torch', torch.__version__, 'sees GPU:', ok); sys.exit(0 if ok else 3)" || { echo "[job] no usable GPU on $(hostname)"; exit 3; }
echo "[job] $(date '+%F %T') start: exp8/src/e8run.py e4_eval --names P2-clean__G__s0 P2-clean__G__s1 P2-clean__N__s0 P2-clean__N__s1 P2-clean__R0__s0 P2-clean__R0__s1 P2-clean__T__s0 P2-clean__T__s1"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I exp8/src/e8run.py e4_eval --names P2-clean__G__s0 P2-clean__G__s1 P2-clean__N__s0 P2-clean__N__s1 P2-clean__R0__s0 P2-clean__R0__s1 P2-clean__T__s0 P2-clean__T__s1 || { echo "[job] FAILED: exp8/src/e8run.py"; exit 1; }
echo "[job] $(date '+%F %T') start: exp8/src/e8_eval2.py answers --names P2-clean__none__s0 P2-clean__none__s1 P2-clean__G__s0 P2-clean__G__s1 P2-clean__N__s0 P2-clean__N__s1 P2-clean__R0__s0 P2-clean__R0__s1 P2-clean__T__s0 P2-clean__T__s1"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I exp8/src/e8_eval2.py answers --names P2-clean__none__s0 P2-clean__none__s1 P2-clean__G__s0 P2-clean__G__s1 P2-clean__N__s0 P2-clean__N__s1 P2-clean__R0__s0 P2-clean__R0__s1 P2-clean__T__s0 P2-clean__T__s1 || { echo "[job] FAILED: exp8/src/e8_eval2.py"; exit 1; }
echo "[job] $(date '+%F %T') start: exp8/src/e8run.py e4_eval_post:items"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I exp8/src/e8run.py e4_eval_post:items || { echo "[job] FAILED: exp8/src/e8run.py"; exit 1; }
echo "[job] $(date '+%F %T') start: exp8/src/e8_eval2.py items"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I exp8/src/e8_eval2.py items || { echo "[job] FAILED: exp8/src/e8_eval2.py"; exit 1; }
echo "[job] $(date '+%F %T') start: exp8/src/e8run.py e2_judge items exp8/outputs/judge/all_items.jsonl exp8/outputs/judge/replies.jsonl"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I exp8/src/e8run.py e2_judge items exp8/outputs/judge/all_items.jsonl exp8/outputs/judge/replies.jsonl || { echo "[job] FAILED: exp8/src/e8run.py"; exit 1; }
echo "[job] $(date '+%F %T') all commands finished"
