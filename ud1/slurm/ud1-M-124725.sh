#!/bin/bash
#SBATCH --job-name=ud1-M
#SBATCH --partition=<anonymous>
#SBATCH --account=<anonymous>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --exclude=<anonymous>
#SBATCH --output=/home/u4161854/loracle_em_exp1/ud1/logs/slurm-%x-%j.out
export HF_HOME=/home/u4161854/loracle_em_exp1/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export XDG_CACHE_HOME=/home/u4161854/loracle_em_exp1/env/xdg_cache
export TMPDIR=/home/u4161854/loracle_em_exp1/ud1/tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd /home/u4161854/loracle_em_exp1
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT"
df -BG /home/u4161854 /work/u4161854 | tail -2
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -B -c "import torch,sys; ok=torch.cuda.is_available(); print('[job] torch', torch.__version__, 'sees GPU:', ok); sys.exit(0 if ok else 3)" || { echo "[job] FAILED: no usable GPU on $(hostname)"; exit 3; }
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -B -c "import torch,sys; f,t=torch.cuda.mem_get_info(); u=(t-f)/2**30; print('[job] GPU memory already in use: %.1f GB' % u); sys.exit(0 if u < 4 else 3)" || { echo "[job] FAILED: the GPU on $(hostname) is in use by another process"; exit 3; }
echo "[job] $(date '+%F %T') start: pb2/src/pb2_parallel.py UM :: ud1/src/ud1_measure.py base K-every K-adv1 P0 P0-s2 K-codeL P0@20 P0@40 P0@100 C1r@5 C1r@10 U-every@5 U-every@10 U-every@20 U-every@40 U-every@80 U-every@150 :: ud1/src/ud1_measure.py --wait-base U-adv1@5 U-adv1@10 U-adv1@20 U-adv1@40 U-adv1@80 U-adv1@150 C3@5 C3@10 C3@20 C3@40 C3@80 C3@150 C2@40 C2@150 rel:every rel:adv1"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -B pb2/src/pb2_parallel.py UM :: ud1/src/ud1_measure.py base K-every K-adv1 P0 P0-s2 K-codeL P0@20 P0@40 P0@100 C1r@5 C1r@10 U-every@5 U-every@10 U-every@20 U-every@40 U-every@80 U-every@150 :: ud1/src/ud1_measure.py --wait-base U-adv1@5 U-adv1@10 U-adv1@20 U-adv1@40 U-adv1@80 U-adv1@150 C3@5 C3@10 C3@20 C3@40 C3@80 C3@150 C2@40 C2@150 rel:every rel:adv1 || { echo "[job] FAILED: pb2/src/pb2_parallel.py"; exit 1; }
echo "[job] $(date '+%F %T') all commands finished"
