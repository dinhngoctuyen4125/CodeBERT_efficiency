#!/bin/bash

#SBATCH --job-name=codebert
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G

# Chạy trực tiếp bằng bash: ghim GPU 0.
# Chạy bằng sbatch: để Slurm cấp phát, ghim tay sẽ đè lên phân cấp của nó và
# có thể đâm vào GPU của job khác.
if [ -z "$SLURM_JOB_ID" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

echo "job=${SLURM_JOB_ID:-none} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

/home/ritsu/miniconda3/envs/codebert/bin/python train_codebert.py --fp16