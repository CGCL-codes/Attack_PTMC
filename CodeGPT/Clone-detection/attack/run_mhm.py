import os

os.system("CUDA_VISIBLE_DEVICES=0 python attack_mhm.py \
    --output_dir=../saved_models \
    --model_type=gpt2 \
    --tokenizer_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --model_name_or_path=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --config_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --csv_store_path result/attack_mhm_all.csv \
    --original\
    --eval_data_file=../../../dataset/Clone-detection/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 2 \
    --seed 123456  2>&1")

