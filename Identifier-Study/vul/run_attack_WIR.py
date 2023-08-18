import os

os.system("CUDA_VISIBLE_DEVICES=0 python attack_WIR.py \
    --output_dir=../../CodeBERT/Vulnerability-detection2/saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --block_size 512 \
    --statement Throw \
    --eval_batch_size 2 \
    --seed 123456  2>&1")

