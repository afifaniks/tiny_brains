#!/bin/bash
#SBATCH -J tiny_brains
#SBATCH -t :00:00
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH --partition=disa --gres=gpu:1
#SBATCH --output=./slurm_out/%j.out

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate tiny_brains && \
jupyter notebook --no-browser --ip="*" --port=9779 --NotebookApp.token=letmepass777