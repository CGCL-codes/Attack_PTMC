import os

os.system("CUDA_VISIBLE_DEVICES=2 python attack_MHM.py \
    --output_dir=../../CodeBERT/Clone-detection/saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --base_model=microsoft/codebert-base-mlm \
    --statement If \
    --block_size 512 \
    --eval_batch_size 2 \
    --seed 123456  2>&1")