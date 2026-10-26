#!/bin/bash
#SBATCH --job-name=lem23a-D1
#SBATCH --partition=dev
#SBATCH --account=MST113101
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:12:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn007,25a-hgpn027,25a-hgpn145,25a-hgpn148,25a-hgpn004
#SBATCH --output=/home/u4161854/loracle_em_exp1/exp23/logs/slurm-%x-%j.out

export HF_HOME=/home/u4161854/loracle_em_exp1/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export XDG_CACHE_HOME=/home/u4161854/loracle_em_exp1/env/xdg_cache
export TMPDIR=/home/u4161854/loracle_em_exp1/exp23/tmp
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
cd /home/u4161854/loracle_em_exp1
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -c "import torch,sys; ok=torch.cuda.is_available(); print('[job] torch', torch.__version__, 'sees GPU:', ok); sys.exit(0 if ok else 3)" || { echo "[job] FAILED: no GPU"; exit 3; }
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -c "import torch,sys; f,t=torch.cuda.mem_get_info(); u=(t-f)/2**30; print('[job] GPU memory already in use: %.1f GB' % u); sys.exit(0 if u < 4 else 3)" || { echo "[job] FAILED: GPU in use"; exit 3; }
df -h /home/u4161854 /work/u4161854
echo "[job] $(date '+%F %T') start: exp23/src/e23_parallel.py D1 :: exp23/src/e23_measure.py base P0@20 P0@40 P0@100 :: exp23/src/e23_sample.py --a-rand-cells P0@20 P0@40 P0@100"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I exp23/src/e23_parallel.py D1 :: exp23/src/e23_measure.py base P0@20 P0@40 P0@100 :: exp23/src/e23_sample.py --a-rand-cells P0@20 P0@40 P0@100 || { echo "[job] FAILED: exp23/src/e23_parallel.py"; exit 1; }
echo "[job] $(date '+%F %T') start: exp23/src/e23_judge.py items dose P0@20 P0@40 P0@100"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I exp23/src/e23_judge.py items dose P0@20 P0@40 P0@100 || { echo "[job] FAILED: exp23/src/e23_judge.py"; exit 1; }
echo "[job] $(date '+%F %T') start: exp23/src/e23_judge.py judge dose"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I exp23/src/e23_judge.py judge dose || { echo "[job] FAILED: exp23/src/e23_judge.py"; exit 1; }
echo "[job] $(date '+%F %T') all commands finished"
