import os

os.system("CUDA_VISIBLE_DEVICES=3 python attack_WIR.py \
    --output_dir=../../CodeBERT/Code-summarization/saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --base_model=microsoft/codebert-base-mlm \
    --statement If \
    --block_size 512 \
    --eval_batch_size 1 \
    --max_source_length 256 \
    --max_target_length 128 \
    --beam_size 10 \
    --seed 123456 2>&1")