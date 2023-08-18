import os

os.system("CUDA_VISIBLE_DEVICES=2 python attack_style.py \
    --output_dir=../saved_models \
    --model_type=gpt2 \
    --model_name_or_path microsoft/CodeGPT-small-java-adaptedGPT2 \
    --dev_filename=../../../dataset/Code-summarization/test_sampled.jsonl \
    --csv_store_path1 result/attack_style_all.csv \
    --load_model_path=../saved_models/checkpoint-best-bleu/pytorch_model.bin \
    --eval_batch_size 2 \
    --max_source_length 256 \
    --max_target_length 128 \
    --beam_size 10 \
    --seed 123456  2>&1")

