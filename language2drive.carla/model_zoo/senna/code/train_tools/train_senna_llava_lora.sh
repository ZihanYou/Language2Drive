#!/bin/bash

MODEL_PATH="/path/to/your/model/directory/"
DATA="/path/to/your/data/directory/your_finetune_qa_data.json"
OUT_DIR="/path/to/your/output/directory/"

cd ${WORKING_PATH}

# setup environments
echo "Setup environments..."
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"


# add python envs & data dir to workspace
ln -s /cluster_home/custom_data/data/Senna/data .
ln -s /cluster_home/custom_data/data/Senna/checkpoints .


# specify the python path
export PATH=/path/to/your/python//bin:$PATH


deepspeed llava/senna/train_senna_llava_multi_img.py \
    --deepspeed llava/zero3.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --freeze_img_adapter False \
    --init_weight_img_adapter False \
    --model_name_or_path $MODEL \
    --version v1 \
    --data_path $DATA \
    --vision_tower /path/to/clip-vit-large-patch14-336/ \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 3 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
