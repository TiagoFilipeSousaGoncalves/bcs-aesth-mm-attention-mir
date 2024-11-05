#!/bin/bash 
# 
#SBATCH --partition=gpu_min11gb  # Debug partition 
#SBATCH --qos=gpu_min11gb_ext         # Debug QoS level 
#SBATCH --job-name=segment    # Job name 
#SBATCH -o slurm.%N.%j.out       # File containing STDOUT output 
#SBATCH -e slurm.%N.%j.err       # File containing STDERR output 

echo "Started job." 

python main_image.py

echo "Finished job."