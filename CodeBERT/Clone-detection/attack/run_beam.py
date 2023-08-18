import os

os.system("CUDA_VISIBLE_DEVICES=0 python attack_beam.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --csv_store_path result/attack_beam_all.csv \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=../../../dataset/Clone-detection/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 2 \
    --beam_size 2 \
    --seed 123456")