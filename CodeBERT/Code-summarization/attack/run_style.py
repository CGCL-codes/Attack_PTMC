import os

os.system("CUDA_VISIBLE_DEVICES=2 python attack_style.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --model_name_or_path microsoft/codebert-base \
    --dev_filename=../../../dataset/Code-summarization/test_sampled.jsonl \
    --csv_store_path1 result/attack_style_all.csv \
    --block_size 512 \
    --eval_batch_size 32 \
    --max_source_length 256 \
    --max_target_length 128 \
    --beam_size 10 \
    --seed 123456  2>&1")

