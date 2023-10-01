#!/bin/bash
#SBATCH --account lcls
#SBATCH --constraint gpu
#SBATCH --qos debug
#SBATCH --time 00:30:00
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH --output=/pscratch/sd/z/zhantao/neurorient_repo/logs/%x.%j.out

export SLURM_CPU_BIND="cores"

module load python
source activate /pscratch/sd/z/zhantao/conda/om

srun python train.py

# perform any cleanup or short post-processing here