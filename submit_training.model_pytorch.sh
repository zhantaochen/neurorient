#!/bin/bash
#SBATCH --job-name=7OK2
#SBATCH --account lcls_g
#SBATCH --constraint gpu
#SBATCH --qos debug
#SBATCH --time 00:29:00
#!SBATCH --qos regular
#!SBATCH --time 12:00:00
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 11
#SBATCH --gpus-per-task 1
#SBATCH --output=slurm/%j.log
#SBATCH --error=slurm/%j.err

export SLURM_CPU_BIND="cores"

## module load python
## source activate /pscratch/sd/z/zhantao/conda/om

module load cray-mpich-abi

python train.model_pytorch.py

# perform any cleanup or short post-processing here
