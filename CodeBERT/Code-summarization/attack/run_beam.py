import os

os.system("CUDA_VISIBLE_DEVICES=1 python attack_beam.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --csv_store_path result/attack_beam_all.csv \
    --eval_data_file=../../../dataset/Code-summarization/test_sampled.jsonl \
    --block_size 512 \
    --eval_batch_size 2 \
    --max_source_length 256 \
    --max_target_length 128 \
    --beam_size 10 \
    --beam_size_num 5 \
    --seed 123456")