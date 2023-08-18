import os

os.system("CUDA_VISIBLE_DEVICES=0 python attack_style.py \
    --output_dir=../saved_models \
    --model_name_or_path ../../checkpoint_11_100000.pt \
    --dev_filename=../../../dataset/Code-summarization/test_sampled.jsonl \
    --load_model_path ../saved_models/checkpoint-best-bleu/pytorch_model.bin \
    --csv_store_path1 result/attack_style_all.csv \
    --tokenizer_name=../../sentencepiece.bpe.model \
    --max_source_length 256 \
    --max_target_length 128 \
    --beam_size 10 \
    --eval_batch_size 2 \
    --seed 123456  2>&1")

