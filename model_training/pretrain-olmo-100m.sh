#!/bin/bash

# Run configuration for modified tinyllama.py with OLMo model
# Using offline WandB and pre-tokenized data

export WANDB_MODE=offline
export WANDB_PROJECT=olmo_pretraining
export WANDB_ENTITY=YOUR_ENTITY
export WANDB_API_KEY=YOUR_KEY

export MODEL_NAME=olmo_100m_n$1
export WANDB_NAME=$MODEL_NAME
export NUMBER_OF_GPUS=1

# Add model_training directory to Python path
export PYTHONPATH=/home/sashi/pinakin/swarm/regmix/model_training:$PYTHONPATH

lightning run model \
    --node-rank=0  \
    --main-address=127.0.0.1 \
    --accelerator=cuda \
    --num-nodes=1 \
    --devices=$NUMBER_OF_GPUS \
    pretrain/tinyllama.py \
    --data_dir /home/sashi/prasanjith/LLAMA-3.2/Hawa \
    --out_name $MODEL_NAME \
    --data_seed 3406 \
    --resume False
