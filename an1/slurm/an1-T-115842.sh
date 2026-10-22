#!/bin/bash
#SBATCH --job-name=an1-T
#SBATCH --partition=<anonymous>
#SBATCH --account=<anonymous>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:48:00
#SBATCH --exclude=<anonymous>
#SBATCH --output=/home/u4161854/loracle_em_exp1/an1/logs/slurm-%x-%j.out
export HF_HOME=/home/u4161854/loracle_em_exp1/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export XDG_CACHE_HOME=/home/u4161854/loracle_em_exp1/env/xdg_cache
export TMPDIR=/home/u4161854/loracle_em_exp1/an1/tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd /home/u4161854/loracle_em_exp1
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT"
df -BG /home/u4161854 /work/u4161854 | tail -2
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -B -c "import torch,sys; ok=torch.cuda.is_available(); print('[job] torch', torch.__version__, 'sees GPU:', ok); sys.exit(0 if ok else 3)" || { echo "[job] FAILED: no usable GPU on $(hostname)"; exit 3; }
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -B -c "import torch,sys; f,t=torch.cuda.mem_get_info(); u=(t-f)/2**30; print('[job] GPU memory already in use: %.1f GB' % u); sys.exit(0 if u < 4 else 3)" || { echo "[job] FAILED: the GPU on $(hostname) is in use by another process"; exit 3; }
echo "[job] $(date '+%F %T') start: pb2/src/pb2_parallel.py AT :: an1/src/an1_train.py Rlt-F :: an1/src/an1_train.py Rpet-F"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -B pb2/src/pb2_parallel.py AT :: an1/src/an1_train.py Rlt-F :: an1/src/an1_train.py Rpet-F || { echo "[job] FAILED: pb2/src/pb2_parallel.py"; exit 1; }
echo "[job] $(date '+%F %T') start: pb2/src/pb2_parallel.py AE :: an1/src/an1_sample.py Rlt-F Rpet-F Rev-F R0-F base --wait :: an1/src/an1_measure.py Rlt-F Rpet-F R0-F --wait"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -B pb2/src/pb2_parallel.py AE :: an1/src/an1_sample.py Rlt-F Rpet-F Rev-F R0-F base --wait :: an1/src/an1_measure.py Rlt-F Rpet-F R0-F --wait || { echo "[job] FAILED: pb2/src/pb2_parallel.py"; exit 1; }
echo "[job] $(date '+%F %T') all commands finished"
