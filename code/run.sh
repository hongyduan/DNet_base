#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python run.py --max_steps_en 547000 \
    --fl 0 \
    -init_en /storage/hyduan/DNet/DNet_base/save/en \
    --cuda
