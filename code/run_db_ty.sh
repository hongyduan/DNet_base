#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python run.py --max_steps_ty 360 \
    --data_path /storage/hyduan/DNet/DNet_base/data/dbpedia_result \
    --batch_size_ty 512 \
    --test_batch_size_ty 4 \
    --log_steps_ty 10 \
    --test_log_steps_ty 2 \
    --fl 1 \
    --negative_sample_size_ty 128 \
    --gamma_intra 0.5 \
    --hidden_dim_ty 50 \
    --learning_rate_ty 0.001 \
    --save_path_en /storage/hyduan/DNet/DNet_base/save/dbpedia/en \
    --save_path_ty /storage/hyduan/DNet/DNet_base/save/dbpedia/ty \
    --save_checkpoint_steps_ty 180 \
    --cuda
    # -init_ty /storage/hyduan/DNet/DNet_base/save/dbpedia/ty \
