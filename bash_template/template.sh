#!/bin/bash
#SBATCH --account lcls_g
#SBATCH --constraint gpu
#SBATCH --qos regular
#SBATCH --time 10:00:00
#SBATCH --ntasks 4
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH --output=/pscratch/sd/z/zhantao/neurorient_repo/logs/%x.%j.out
#SBATCH --error=/pscratch/sd/z/zhantao/neurorient_repo/logs/%x.%j.err

export SLURM_CPU_BIND="cores"

module load python
conda activate /pscratch/sd/z/zhantao/conda/om

srun python train.py --yaml_file {{ path_yaml }}

# perform any cleanup or short post-processing here
