#!/bin/bash
#SBATCH --job-name=mt1-job2-coin
#SBATCH --partition=dev
#SBATCH --account=MST113101
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:15:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn007,25a-hgpn027,25a-hgpn145,25a-hgpn148,25a-hgpn004,25a-hgpn142
#SBATCH --output=/work/u4161854/loracle_em_exp1/mt1/logs/slurm-%x-%j.out

export HF_HOME=/home/u4161854/mt1_stage/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export XDG_CACHE_HOME=/home/u4161854/mt1_stage/xdg_cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/home/u4161854/mt1_stage/tmp/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
cd /work/u4161854/loracle_em_exp1
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT tmp=$TMPDIR"
echo "- $(date '+%F %H:%M') MT1 job $SLURM_JOB_ID mt1-job2-coin started on $(hostname)" >> /work/u4161854/loracle_em_exp1/RUNLOG.md
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/work/u4161854/loracle_em_exp1/env/mt1_infer/bin/python -I -c "import torch,sys; ok=torch.cuda.is_available(); print('[job] torch', torch.__version__, 'sees GPU:', ok); sys.exit(0 if ok else 3)" || { echo "[job] FAILED: no usable GPU on $(hostname)"; exit 3; }
/work/u4161854/loracle_em_exp1/env/mt1_infer/bin/python -I -c "import torch,sys; f,t=torch.cuda.mem_get_info(); u=(t-f)/2**30; print('[job] GPU memory already in use: %.1f GB' % u); sys.exit(0 if u < 4 else 3)" || { echo "[job] FAILED: the GPU on $(hostname) is in use by another process"; exit 3; }
echo "[job] $(date '+%F %T') start: env/mt1_infer/bin/python -I mt1/src/mt1_job2.py --arm coin --b9-count language"
env/mt1_infer/bin/python -I mt1/src/mt1_job2.py --arm coin --b9-count language || { echo "[job] FAILED: env/mt1_infer/bin/python"; echo "- $(date '+%F %H:%M') MT1 job $SLURM_JOB_ID mt1-job2-coin FAILED in: env/mt1_infer/bin/python -I mt1/src/mt1_job2.py --arm coin --b9-count language" >> /work/u4161854/loracle_em_exp1/RUNLOG.md; exit 1; }
echo "[job] $(date '+%F %T') all commands finished"
echo "- $(date '+%F %H:%M') MT1 job $SLURM_JOB_ID mt1-job2-coin finished OK on $(hostname)" >> /work/u4161854/loracle_em_exp1/RUNLOG.md
rm -rf "$TMPDIR"
