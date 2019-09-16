#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python run.py --max_steps_ty 1500 \
    --data_path /storage/hyduan/DNet/DNet_base/data/dbpedia_result \
    --batch_size_ty 512 \
    --test_batch_size_ty 4 \
    --log_steps_ty 100 \
    --test_log_steps_ty 2 \
    --fl 1 \
    --save_path_en /storage/hyduan/DNet/DNet_base/save/dbpedia/en \
    --save_path_ty /storage/hyduan/DNet/DNet_base/save/dbpedia/ty \
    --save_checkpoint_steps_ty 150 \
    --cuda
    # -init_ty /storage/hyduan/DNet/DNet_base/save/dbpedia/ty \
    
