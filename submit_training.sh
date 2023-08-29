#!/bin/bash
#SBATCH --account lcls
#SBATCH --constraint gpu
#SBATCH --qos regular
#SBATCH --time 12:00:00
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 128
#SBATCH --gpus-per-task 1
#SBATCH --output=/pscratch/sd/z/zhantao/neurorient_repo/model/slurm_logs/%x.%j.out

export SLURM_CPU_BIND="cores"

module load python
source activate /pscratch/sd/z/zhantao/conda/om

srun python training.py

# perform any cleanup or short post-processing here