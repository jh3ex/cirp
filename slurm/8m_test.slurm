#!/usr/bin/env bash
#SBATCH -J pytorchtest
#SBATCH -o pytorchtest-%A.out
#SBATCH -e pytorchtest-%A.err
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -c 1
#SBATCH -t 20:00:00
#SBATCH -A isl_uva

module purge
module load singularity/3.5.2

# Assuming that the container has been copied to the user's home directory
# First config is for algs, and second for envs
singularity run --nv /scratch/jh3ex/pytorch-1.4.0-py37.sif /scratch/jh3ex/cirp/src/main.py --config=trans_vdn_a2c --env-config=production_more_machines
