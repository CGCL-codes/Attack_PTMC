import os

os.system("CUDA_VISIBLE_DEVICES=3 python attack_style.py \
    --output_dir=../saved_models \
    --model_name_or_path=../../checkpoint_11_100000.pt \
    --tokenizer_name=../../sentencepiece.bpe.model \
    --base_model=microsoft/codebert-base-mlm \
    --csv_store_path1 result/attack_style_all.csv \
    --use_ga \
    --eval_data_file=../../../dataset/Clone-detection/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 2 \
    --seed 123456  2>&1")

