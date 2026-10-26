#!/bin/bash
#SBATCH --job-name=ls1-J0
#SBATCH --partition=dev
#SBATCH --account=MST113101
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:48:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn007,25a-hgpn027,25a-hgpn145,25a-hgpn148,25a-hgpn004
#SBATCH --output=/home/u4161854/loracle_em_exp1/ls1/logs/slurm-%x-%j.out
export HF_HOME=/home/u4161854/loracle_em_exp1/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
mkdir -p /home/u4161854/loracle_em_exp1/ls1/systmp
export XDG_CACHE_HOME=/home/u4161854/loracle_em_exp1/env/xdg_cache
export TMPDIR=/home/u4161854/loracle_em_exp1/ls1/systmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/u4161854/loracle_em_exp1
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT"
df -BG /home/u4161854 /work/u4161854 | tail -2
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -B -c "import torch,sys; ok=torch.cuda.is_available(); print('[job] torch', torch.__version__, 'sees GPU:', ok); sys.exit(0 if ok else 3)" || { echo "[job] FAILED: no usable GPU on $(hostname)"; exit 3; }
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -B -c "import torch,sys; f,t=torch.cuda.mem_get_info(); u=(t-f)/2**30; print('[job] GPU memory already in use: %.1f GB' % u); sys.exit(0 if u < 4 else 3)" || { echo "[job] FAILED: the GPU on $(hostname) is in use by another process"; exit 3; }
echo "[job] $(date '+%F %T') start: ls1/src/ls1_parallel.py J0 :: ls1/src/ls1_train.py replay-K :: ls1/src/ls1_train.py replay-R ++ ls1/src/ls1_phase0.py ++ ls1/src/ls1_judge.py check ++ ls1/src/ls1_eval.py base:jf,rjf P0:jf K-every:jf,agrid R0-F:rjf Rev-F:rjf"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -B ls1/src/ls1_parallel.py J0 :: ls1/src/ls1_train.py replay-K :: ls1/src/ls1_train.py replay-R ++ ls1/src/ls1_phase0.py ++ ls1/src/ls1_judge.py check ++ ls1/src/ls1_eval.py base:jf,rjf P0:jf K-every:jf,agrid R0-F:rjf Rev-F:rjf || { echo "[job] FAILED: ls1/src/ls1_parallel.py"; exit 1; }
echo "[job] $(date '+%F %T') all commands finished"
