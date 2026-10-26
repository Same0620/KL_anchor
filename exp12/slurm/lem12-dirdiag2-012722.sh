#!/bin/bash
#SBATCH --job-name=lem12-dirdiag2
#SBATCH --partition=dev
#SBATCH --account=MST113101
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:18:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn007,25a-hgpn027,25a-hgpn145,25a-hgpn148
#SBATCH --output=/home/u4161854/loracle_em_exp1/exp12/logs/slurm-%x-%j.out

export HF_HOME=/home/u4161854/loracle_em_exp1/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export XDG_CACHE_HOME=/home/u4161854/loracle_em_exp1/env/xdg_cache
export TMPDIR=/home/u4161854/loracle_em_exp1/exp12/tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd /home/u4161854/loracle_em_exp1
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -c "import torch,sys; ok=torch.cuda.is_available(); print('[job] torch', torch.__version__, 'sees GPU:', ok); sys.exit(0 if ok else 3)" || { echo "[job] FAILED: no usable GPU on $(hostname)"; exit 3; }
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -c "import torch,sys; f,t=torch.cuda.mem_get_info(); u=(t-f)/2**30; print('[job] GPU memory already in use: %.1f GB' % u); sys.exit(0 if u < 4 else 3)" || { echo "[job] FAILED: the GPU on $(hostname) is in use by another process"; exit 3; }
echo "[job] $(date '+%F %T') start: exp12/src/e12_dirdiag2.py --smoke"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I exp12/src/e12_dirdiag2.py --smoke || { echo "[job] FAILED: exp12/src/e12_dirdiag2.py"; exit 1; }
echo "[job] $(date '+%F %T') start: exp12/src/e12_dirdiag2.py"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I exp12/src/e12_dirdiag2.py || { echo "[job] FAILED: exp12/src/e12_dirdiag2.py"; exit 1; }
echo "[job] $(date '+%F %T') all commands finished"
