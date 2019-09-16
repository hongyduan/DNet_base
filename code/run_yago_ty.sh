#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python run.py --max_steps_ty 70000 \
    --data_path /storage/hyduan/DNet/DNet_base/data/yago_result \
    --batch_size_ty 100 \
    --test_batch_size_ty 4 \
    --log_steps_ty 200 \
    --test_log_steps_ty 200 \
    --fl 1 \
    --save_path_en /storage/hyduan/DNet/DNet_base/save/yago/en \
    --save_path_ty /storage/hyduan/DNet/DNet_base/save/yago/ty \
    --save_checkpoint_steps_ty 7000 \
    --cuda
    # init_ty /storage/hyduan/DNet/DNet_base/save/yago/ty
