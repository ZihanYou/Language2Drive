#!/bin/bash
BASE_PORT=2020
BASE_TM_PORT=8020
IS_BENCH2DRIVE=True
export WORK_DIR=/home/karenz/coding/simlingo_cleanup
BASE_ROUTES=/home/karenz/coding/simlingo_cleanup/leaderboard/data/bench2drive220
TEAM_AGENT=team_code/agent_simlingo_base.py
TEAM_CONFIG=/home/karenz/coding/simlingo_cleanup/outputs/2025_04_07_05_12_40_simlingo_basev2_after_cleanup/checkpoints/epoch=029.ckpt/pytorch_model.pt
# TEAM_CONFIG=/home/katrinrenz/coding/wayve_carla/outputs/cvpr/2024_10_29_02_59_56_AML_305_15_internvl1B_lmdriveaugm_commentary_cotv3_vqav3_dpoalignv3_safetyflag_alltowns/checkpoints/epoch=013.ckpt/pytorch_model.pt
# TEAM_CONFIG=/home/katrinrenz/coding/wayve_carla/outputs/204_XX/2024_08_29_08_37_19_AML_204_00_internvl1B_lmdriveaugm_route_as_target_point_command/checkpoints/epoch=029.ckpt/pytorch_model.pt
# TEAM_CONFIG=your_team_agent_config.py+your_team_agent_ckpt.pth # for UniAD and VAD
BASE_CHECKPOINT_ENDPOINT=simlingo
SAVE_PATH=./simlingo
PLANNER_TYPE=only_traj

mkdir -p $SAVE_PATH

GPU_RANK=0
PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}.json"

bash Bench2Drive/leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK
