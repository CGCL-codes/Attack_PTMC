import os

os.system("CUDA_VISIBLE_DEVICES=2 python attack_wir_random.py \
    --output_dir=../saved_models \
    --model_name_or_path=../../checkpoint_11_100000.pt \
    --tokenizer_name=../../sentencepiece.bpe.model \
    --csv_store_path result/attack_wir_all.csv \
    --eval_data_file=../../../dataset/Clone-detection/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 2 \
    --seed 123456")

