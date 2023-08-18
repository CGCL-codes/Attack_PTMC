import os

os.system("CUDA_VISIBLE_DEVICES=1 python attack_beam.py \
    --output_dir=../saved_models \
    --model_type=gpt2 \
    --model_name_or_path microsoft/CodeGPT-small-java-adaptedGPT2 \
    --csv_store_path result/attack_beam_all.csv \
    --eval_data_file=../../../dataset/Code-summarization/test_sampled.jsonl \
    --load_model_path=../saved_models/checkpoint-best-bleu/pytorch_model.bin \
    --eval_batch_size 2 \
    --block_size 512 \
    --max_source_length 256 \
    --max_target_length 128 \
    --beam_size 10 \
    --beam_size_num 5 \
    --seed 123456")