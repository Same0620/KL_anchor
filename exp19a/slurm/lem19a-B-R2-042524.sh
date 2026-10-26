#!/bin/bash
#SBATCH --job-name=lem19a-B-R2
#SBATCH --partition=dev
#SBATCH --account=MST113101
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:45:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn007,25a-hgpn027,25a-hgpn145,25a-hgpn148,25a-hgpn004
#SBATCH --output=/home/u4161854/loracle_em_exp1/exp19a/logs/slurm-%x-%j.out

export HF_HOME=/home/u4161854/loracle_em_exp1/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export XDG_CACHE_HOME=/home/u4161854/loracle_em_exp1/env/xdg_cache
export TMPDIR=/home/u4161854/loracle_em_exp1/exp19a/tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd /home/u4161854/loracle_em_exp1
echo "[job] $(date '+%F %T') node=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -c "import torch,sys; ok=torch.cuda.is_available(); print('[job] torch', torch.__version__, 'sees GPU:', ok); sys.exit(0 if ok else 3)" || { echo "[job] FAILED: no usable GPU on $(hostname)"; exit 3; }
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I -c "import torch,sys; f,t=torch.cuda.mem_get_info(); u=(t-f)/2**30; print('[job] GPU memory already in use: %.1f GB' % u); sys.exit(0 if u < 4 else 3)" || { echo "[job] FAILED: the GPU on $(hostname) is in use by another process"; exit 3; }
echo "[job] $(date '+%F %T') start: exp19a/src/e19_jobB.py --readers R2 --models em-risky_financial-broad em-risky_financial-narrow em-bad_medical-broad em-bad_medical-narrow em-extreme_sports-broad em-extreme_sports-narrow em-bad_legal-broad em-bad_legal-narrow em-unsafe_diy-broad em-unsafe_diy-narrow em-reckless_driving-broad em-reckless_driving-narrow em-bad_beauty-broad em-bad_beauty-narrow em-bad_crypto-broad em-bad_crypto-narrow em-bad_dental-broad em-bad_dental-narrow em-bad_gambling-broad em-bad_gambling-narrow em-bad_heat-broad em-bad_heat-narrow em-bad_nutrition-broad em-bad_nutrition-narrow em-bad_pesticides-broad em-bad_pesticides-narrow em-bad_petcare-broad em-bad_petcare-narrow em-bad_rehab-broad em-bad_rehab-narrow em-bad_tattoo-broad em-bad_tattoo-narrow em-bad_tenancy-broad em-bad_tenancy-narrow em-unsafe_boating-broad em-unsafe_boating-narrow em-unsafe_camping-broad em-unsafe_camping-narrow em-unsafe_chemicals-broad em-unsafe_chemicals-narrow em-unsafe_fire-broad em-unsafe_fire-narrow em-unsafe_firearms-broad em-unsafe_firearms-narrow em-unsafe_scuba-broad em-unsafe_scuba-narrow T-normal T-KL C-normal-s0@40 C-normal-s1 C-normal-s2 C-KL-s1"
/home/u4161854/loracle_em_exp1/env/venv/bin/python -I exp19a/src/e19_jobB.py --readers R2 --models em-risky_financial-broad em-risky_financial-narrow em-bad_medical-broad em-bad_medical-narrow em-extreme_sports-broad em-extreme_sports-narrow em-bad_legal-broad em-bad_legal-narrow em-unsafe_diy-broad em-unsafe_diy-narrow em-reckless_driving-broad em-reckless_driving-narrow em-bad_beauty-broad em-bad_beauty-narrow em-bad_crypto-broad em-bad_crypto-narrow em-bad_dental-broad em-bad_dental-narrow em-bad_gambling-broad em-bad_gambling-narrow em-bad_heat-broad em-bad_heat-narrow em-bad_nutrition-broad em-bad_nutrition-narrow em-bad_pesticides-broad em-bad_pesticides-narrow em-bad_petcare-broad em-bad_petcare-narrow em-bad_rehab-broad em-bad_rehab-narrow em-bad_tattoo-broad em-bad_tattoo-narrow em-bad_tenancy-broad em-bad_tenancy-narrow em-unsafe_boating-broad em-unsafe_boating-narrow em-unsafe_camping-broad em-unsafe_camping-narrow em-unsafe_chemicals-broad em-unsafe_chemicals-narrow em-unsafe_fire-broad em-unsafe_fire-narrow em-unsafe_firearms-broad em-unsafe_firearms-narrow em-unsafe_scuba-broad em-unsafe_scuba-narrow T-normal T-KL C-normal-s0@40 C-normal-s1 C-normal-s2 C-KL-s1 || { echo "[job] FAILED: exp19a/src/e19_jobB.py"; exit 1; }
echo "[job] $(date '+%F %T') all commands finished"
