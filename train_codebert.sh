#!/bin/bash

#SBATCH --job-name=codebert
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=defq
#SBATCH --qos=short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

/home/ritsu/miniconda3/envs/codebert/bin/python train_codebert.py --fp16