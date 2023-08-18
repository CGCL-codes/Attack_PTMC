import os

os.system("CUDA_VISIBLE_DEVICES=1 python attack_alert.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --model_name_or_path microsoft/codebert-base \
    --csv_store_path result/attack_alert_all.csv \
    --dev_filename=../../../dataset/Code-summarization/test_sampled.jsonl \
    --use_ga \
    --block_size 512 \
    --eval_batch_size 2 \
    --max_source_length 256 \
    --max_target_length 128 \
    --beam_size 10 \
    --seed 123456")

