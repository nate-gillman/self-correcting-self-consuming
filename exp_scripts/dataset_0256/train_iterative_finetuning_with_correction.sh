#!/bin/bash

# feel free to use whatever seed you want
SEED=42

# set these if you want to use wandb
USE_WANDB=0 # (0, 1) --> (False, True)
WANDB_PROJECT_NAME="dataset_0256"
WANDB_EXP_NAME="iterative_finetuning_with_correction_${1}"

# if you run into GL issues, try deleting this line
export PYOPENGL_PLATFORM=''

# don't change these
MINI_DATASET_DIR="dataset/HumanML3D_subset_0256"
SAVE_DIR="exp_outputs/dataset_0256/synthetic_percent_${1}_iterative_finetuning_with_correction"
SYNTH_AUG_PERCENT=$(echo "scale=2; ${1} / 100" | bc -l | awk '{printf "%.2f", $0}')

for generation in {1..50}
do 

  # Run the script to synthesize examples
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
    --num_steps 1000 \
    --save_interval 2000 \
    --overwrite \
    --synthetic_augmentation_percent ${SYNTH_AUG_PERCENT} \
    --augmentation_type imitation_output \
    --uhc_model_option uhc_explicit \
    --is_self_consuming True \
    --start_at_generation $generation \
    --overwrite \
    --ONLY_SYNTHESIZE_EXAMPLES \
    --seed $SEED

  # Run the script to train
  python -m train.train_mdm_iterative_finetuning\
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
    --num_steps 1000 \
    --save_interval 2000 \
    --overwrite \
    --synthetic_augmentation_percent ${SYNTH_AUG_PERCENT} \
    --augmentation_type imitation_output \
    --uhc_model_option uhc_explicit \
    --is_self_consuming True \
    --start_at_generation $generation \
    --overwrite \
    --seed $SEED

done