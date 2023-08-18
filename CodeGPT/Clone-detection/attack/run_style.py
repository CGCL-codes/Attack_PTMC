import os

os.system("CUDA_VISIBLE_DEVICES=1 python attack_style.py \
    --output_dir=../saved_models \
    --model_type=gpt2 \
    --tokenizer_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --model_name_or_path=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --base_model=microsoft/codebert-base-mlm \
    --csv_store_path1 result/attack_style_all.csv \
    --config_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --eval_data_file=../../../dataset/Clone-detection/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 2 \
    --seed 123456  2>&1")

