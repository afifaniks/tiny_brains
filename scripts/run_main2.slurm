#!/bin/bash
#SBATCH -J tiny_brains
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --mem=80G
#SBATCH --partition=gpu-a100 --gres=gpu:1
#SBATCH --output=./slurm_out/%j.out

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate tiny_brains && \
python main2.py