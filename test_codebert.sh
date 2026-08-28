#!/bin/bash

#SBATCH --job-name=codebert-test
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --time=0:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G

export CUDA_VISIBLE_DEVICES=0

/home/ritsu/miniconda3/envs/codebert/bin/python test_codebert.py --fp16
