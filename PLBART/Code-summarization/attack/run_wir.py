import os

os.system("CUDA_VISIBLE_DEVICES=1 python attack_wir.py \
    --output_dir=../saved_models \
    --model_name_or_path ../../checkpoint_11_100000.pt \
    --load_model_path ../saved_models/checkpoint-best-bleu/pytorch_model.bin \
    --tokenizer_name=../../sentencepiece.bpe.model \
    --csv_store_path result/attack_wir_all.csv \
    --dev_filename=../../../dataset/Code-summarization/test_sampled.jsonl \
    --eval_batch_size 2 \
    --max_source_length 256 \
    --max_target_length 128 \
    --beam_size 10 \
    --seed 123456")

