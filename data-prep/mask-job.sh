#!/bin/bash

# Set job requirements
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 4:00:00

# Load modules
module load 2022
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
#module load Python/3.10.4-GCCcore-11.3.0

python -u mask-images.py
