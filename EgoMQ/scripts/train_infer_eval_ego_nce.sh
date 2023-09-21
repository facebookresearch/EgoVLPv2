#!/bin/sh
#SBATCH -N 1
#SBATCH --job-name=MAD_RANDOM
#SBATCH --time=5:59:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=90GB
#SBATCH --gres=gpu:4
#SBATCH --chdir=/ibex/ai/home/soldanm/projects/nips22/episodic-memory/MQ/
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0

module load gcc

echo "######################### SLURM JOB ########################"
echo HOST NAME
echo `hostname`
echo "############################################################"

environment=ego
conda_root=$HOME/anaconda3

source $conda_root/etc/profile.d/conda.sh
conda activate $environment

set -ex

# ------------------------ need not change -----------------------------------

FEATURES=<Path_of_Folder_Containing_Pre-extracted_Features>
CHECKPOINT=./outputs/
INPUT_DIM=4096
EPOCHS=50

#BSIZE=32,16,8
#LR_RATE=0.0001, 0.0005, 0.00005, 0.00001
#STEP=15,30,5
#GAMMA=0.1, 0.5, 0.25, 0.05

for BSIZE in 32 16 8
do
	for LR_RATE in 0.0001 0.0005 0.00005 0.00001
	do
		for STEP in 15 30 5
		do
			for GAMMA in 0.05 0.1 0.5 0.25
			do
														                                    
				python Train.py --use_xGPN --is_train true --dataset ego4d --feature_path $FEATURES --checkpoint_path $CHECKPOINT --batch_size $BSIZE --train_lr $LR_RATE --step_size $STEP --step_gamma $GAMMA --input_feat_dim $INPUT_DIM --num_epoch $EPOCHS 
				python Infer.py --use_xGPN --is_train false --dataset ego4d --feature_path $FEATURES --checkpoint_path $CHECKPOINT --input_feat_dim $INPUT_DIM --output_path $CHECKPOINT

				python Eval.py --dataset ego4d --output_path $CHECKPOINT --out_prop_map true --eval_stage all > ${CHECKPOINT}_results_${BSIZE}_${LR_RATE}_${STEP}_${GAMMA}.txt

			done
		done
	done
done	
