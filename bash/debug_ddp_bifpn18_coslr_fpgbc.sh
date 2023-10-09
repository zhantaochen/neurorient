#!/bin/bash
#SBATCH --account lcls
#SBATCH --constraint gpu
#SBATCH --qos debug
#SBATCH --time 00:05:00
#SBATCH --ntasks 4
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH --output=/pscratch/sd/z/zhantao/neurorient_repo/logs/%x.%j.out

export SLURM_CPU_BIND="cores"

module load python
source activate /pscratch/sd/z/zhantao/conda/om

srun python train.py --yaml_file yaml/config_bifpn18_coslr_fpgbc.yaml

# perform any cleanup or short post-processing here