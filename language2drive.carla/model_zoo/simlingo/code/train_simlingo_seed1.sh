#!/bin/bash
#SBATCH --job-name=slv2
#SBATCH --nodes=1
#SBATCH --time=3-00:00
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --output=/YOUR_PATH/results/logs/slv2_%a_%A.out  # File to which STDOUT will be written
#SBATCH --error=/YOUR_PATH/results/logs/slv2_%a_%A.err   # File to which STDERR will be written
#SBATCH --partition=YOUR_PARTITION

# print info about current job
scontrol show job $SLURM_JOB_ID

source ~/.bashrc
conda activate ~/software/anaconda3/envs/simlingo

pwd
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export WORK_DIR=~/coding/07_simlingo/simlingo_cleanup
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}

export MASTER_ADDR=localhost
export NCCL_DEBUG=INFO

export OMP_NUM_THREADS=64 # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
WANDB__SERVICE_WAIT=300 python simlingo_training/train.py experiment=simlingo_seed1 data_module.batch_size=8 gpus=8 name=simlingo_seed1