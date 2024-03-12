#!/bin/bash

# A batch script for running a job on Oscar's 3090 condo, using the Slurm scheduler
# The 3090 condo runs NVIDIA's GeForce RTX 3090 graphics card

#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --constraint=a6000|geforce3090
#SBATCH --exclude=gpu2108,gpu2114,gpu2115,gpu2116
#SBATCH -N 1 # gives one node, makes sure cpu cores are on same node
#SBATCH -c 4 # requests four CPU cores
#SBATCH --mem=63G
#SBATCH -t 96:00:00
#SBATCH -e exp_outputs/slurm_logs/%j.err
#SBATCH -o exp_outputs/slurm_logs/%j.out

# SET UP COMPUTING ENV
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/$USER/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

module load mesa
module load boost/1.80.0
module load patchelf
module load glew
module load cuda
module load ffmpeg

# Activate virtual environment
# Load anaconda module, and other modules
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate /gpfs/home/ngillman/.conda/envs/scsc

# Move to correct working directory
HOME_DIR=/oscar/data/superlab/users/nates_stuff/self-correcting-self-consuming-cleanup
cd ${HOME_DIR}

# put the script execution statement here
rsync -av --exclude='.git' /oscar/data/superlab/users/nates_stuff/self-correcting-self-consuming-cleanup/ /oscar/data/superlab/users/nates_stuff/self-correcting-self-consuming/