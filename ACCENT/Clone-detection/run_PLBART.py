import os
#
os.system("python substitution_PLBART.py \
    --output_dir=PLBART/Clone_detection/saved_models/ \
    --model_type=bart \
    --model_name_or_path=PLBART/checkpoint_11_100000.pt \
    --tokenizer_name=PLBART/sentencepiece.bpe.model \
    --do_train \
    --train_data_file=../../../dataset/Clone-detection/train.txt \
    --eval_data_file=../../../dataset/Clone-detection/valid.txt \
    --test_data_file=../../../dataset/Clone-detection/test.txt \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 2 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1")

#
# os.system("python ACC_PLBART.py \
#     --output_dir=PLBART/Clone_detection/saved_models/ \
#     --model_type=bart \
#     --model_name_or_path=PLBART/checkpoint_11_100000.pt \
#     --tokenizer_name=PLBART/sentencepiece.bpe.model \
#     --do_train \
#     --train_data_file=../../../dataset/Clone-detection/train.txt \
#     --eval_data_file=../../../dataset/Clone-detection/valid.txt \
#     --test_data_file=../../../dataset/Clone-detection/test.txt \
#     --epoch 2 \
#     --block_size 400 \
#     --train_batch_size 16 \
#     --eval_batch_size 32 \
#     --learning_rate 5e-5 \
#     --max_grad_norm 1.0 \
#     --evaluate_during_training \
#     --seed 123456")
