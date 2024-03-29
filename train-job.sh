#!/bin/bash

# Set job requirements
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 8:00:00
#SBATCH --gpus=1

# Load modules
module load 2022
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0

# Run the script, passing all command line arguments
pip install -r requirements.txt
python -u train.py train
