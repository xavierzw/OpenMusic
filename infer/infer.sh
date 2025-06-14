#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export PYTHONPATH=$(pwd):$PYTHONPATH

# TODO: change the yaml, text list, ckpt path

# for better music quality
python3 infer/infer_mos5.py \
--config_yaml audioldm_train/config/mos_as_token/qa_mdt.yaml \
--list_inference test_prompts/test_prompts.lst \
--reload_from_ckpt /checkpoint-2799.ckpt

# for better overrall quality
# python3 infer_mos4.py \
# --config_yaml audioldm_train/config/mos_as_token/qa_mdt.yaml \
# --list_inference prompts/good_prompts_1.lst \
# --reload_from_ckpt "output/model_checkpoint.ckpt"
