#!/bin/bash
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=saad.ashraf@ucalgary.ca
#SBATCH -J 🙃🙃
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu-l40 --gres=gpu:1
#SBATCH --output=./slurm_out/%j.out

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate tiny_brains && \
python main_test.py