#!/bin/bash

# feel free to use whatever seed you want
SEED=42

# set these if you want to use wandb
USE_WANDB=0 # (0, 1) --> (False, True)
WANDB_PROJECT_NAME="dataset_0128"
WANDB_EXP_NAME="generation_0"

# if you run into GL issues, try deleting this line
export PYOPENGL_PLATFORM=''

# don't change these
MINI_DATASET_DIR="dataset/HumanML3D_subset_0128"
SAVE_DIR="exp_outputs/dataset_0128/generation_0"
SYNTH_AUG_PERCENT=0.0
generation=0

# Run the script to train generation 0
python -m train.train_mdm_iterative_finetuning \
  --save_dir ${SAVE_DIR} \
  --use_wandb ${USE_WANDB} \
  --wandb_project_name ${WANDB_PROJECT_NAME} \
  --wandb_exp_name ${WANDB_EXP_NAME} \
  --eval_during_training \
  --eval_rep_times 1 \
  --eval_num_samples 546 \
  --dataset humanml \
  --mini_dataset_dir ${MINI_DATASET_DIR} \
  --filter_dataset_by_keyword NOKEYWORD \
  --num_steps 10000 \
  --save_interval 5000 \
  --overwrite \
  --synthetic_augmentation_percent ${SYNTH_AUG_PERCENT} \
  --is_self_consuming True \
  --start_at_generation $generation \
  --overwrite \
  --seed $SEED

