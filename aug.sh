#!/bin/bash -l

#SBATCH --partition=ssep
#SBATCH --job-name=prune_training_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1,gpu_mem:48000

#SBATCH --partition=sharedp,student,shared

# Activate Conda environment 


python van.py /cig/common04nb/students/deaallay/Waste/rel.jpg /cig/common04nb/students/deaallay/Waste/van.jpg  /cig/common04nb/students/deaallay/Waste