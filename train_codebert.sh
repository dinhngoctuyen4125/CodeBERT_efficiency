#!/bin/bash

#SBATCH --job-name=codebert
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

export CUDA_VISIBLE_DEVICES=0

/home/ritsu/miniconda3/envs/codebert/bin/python train_codebert.py --fp16