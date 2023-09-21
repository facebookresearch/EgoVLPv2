#!/bin/bash
#SBATCH --job-name=EgoVLPv2_Pretraining

#SBATCH --output=/results/slurm_jobs/EgoVLPv2_Pretraining.out
#SBATCH --error=/results/slurm_jobs/EgoVLPv2_Pretraining.err

#SBATCH --partition=egohowto
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --time=14400
#SBATCH --exclusive
#SBATCH --account=all

### init virtual environment if needed
source ~/miniconda3/etc/profile.d/conda.sh
conda activate egovlpv2

srun --label python3 multinode_train_egoclip.py --save_dir /results_EgoVLPv2 --print_freq 100 --config ./configs/pt/egoclip.json

