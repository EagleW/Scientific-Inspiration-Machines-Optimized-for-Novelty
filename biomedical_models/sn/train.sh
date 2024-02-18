#!/bin/bash


export HF_DATASETS_CACHE="data/.hf_dataset_cache"
BASE_MODEL_DIR=../models
BASE_MODEL_NAME=meditron-7b
DATA_ROOT=data/sim_

BASE_MODEL_PATH=$BASE_MODEL_DIR/$BASE_MODEL_NAME
# ====== Config ======
# SEQ_LENGTH=768  # Mistral support up to 32768
SEQ_LENGTH=640  # Mistral support up to 32768

# Learning Rate
# LR=5e-7
LR=2e-6
WEIGHT_DECAY=0.0


# Batch Size
PER_DEVICE_BSZ=1
GRAD_ACC=1

# Steps
N_EPOCH=5
# N_EPOCH=15
RATIO=0.03


echo "Using model from $BASE_MODEL_PATH"
EXP_ID="${BASE_MODEL_NAME}-lr${LR}-seq${SEQ_LENGTH}-ratio${RATIO}"

# - Output Dir
OUTPUT_DIR=models/ckpts/$EXP_ID
export WANDB_PROJECT=hypothesis
export WANDB_NAME=$EXP_ID
# export WANDB_MODE=disabled  # for debug
echo "Saving to $OUTPUT_DIR"


# python \
# CUDA_VISIBLE_DEVICES="0,1" \
accelerate launch \
    --config_file config/fsdp_.yaml \
    trainer.py \
    --train_file $DATA_ROOT/train.json \
    --validation_file $DATA_ROOT/valid.json \
    --model_name_or_path $BASE_MODEL_PATH \
    --with_tracking \
    --output_dir $OUTPUT_DIR \
    --report_to wandb \
    --seed 42 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size $PER_DEVICE_BSZ \
    --gradient_accumulation_steps $GRAD_ACC \
    --num_train_epochs $N_EPOCH \
    --checkpointing_steps "epoch" \
    --lr_scheduler_type linear \
    --learning_rate $LR \
    --weight_decay $WEIGHT_DECAY \
    --lr_scheduler_warmup_ratio $RATIO \
    --block_size $SEQ_LENGTH
    # --checkpointing_steps 1000000 \
    # --no_keep_linebreaks \
    # --resume_from_checkpoint models/ckpts/$EXP_ID/step_800 \
    # --resume_from_checkpoint models/ckpts/$EXP_ID/epoch_4 \
    
    # --config_file config/accelerate.yaml \
    # --checkpointing_steps "epoch" \