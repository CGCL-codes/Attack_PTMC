import os

os.system("CUDA_VISIBLE_DEVICES=0 python run.py \
    --output_dir=../saved_models/ \
    --do_test \
    --model_type roberta \
    --model_name_or_path microsoft/codebert-base \
    --load_model_path ../saved_models/checkpoint-best-bleu/pytorch_model.bin \
    --train_filename=../../../dataset/Code-summarization/train.jsonl \
    --dev_filename=../../../dataset/Code-summarization/valid.jsonl \
    --test_filename=../../../dataset/Code-summarization/test.jsonl \
    --max_source_length 256 \
    --max_target_length 128 \
    --beam_size 10 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    2>&1")
