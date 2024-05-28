#!/bin/bash
#SBATCH -J tiny_brains
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --partition=gpu-a100 --gres=gpu:1
#SBATCH --output=./slurm_out/%j.out

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate tiny_brains && \
python /home/mdafifal.mamun/research/tiny_brains/main.py --data_root /work/disa_lab/projects/tiny_brains