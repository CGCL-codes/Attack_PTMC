# coding=utf-8
# @Time    : 2020/8/13
# @Author  : Zhou Yang
# @Email   : zyang@smu.edu.sg
# @File    : gi_attack.py
'''For attacking CodeBERT models'''
import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append('../code')
sys.path.append('../../../')
sys.path.append('../../../python_parser')
from datetime import datetime
import json
import logging
import argparse
import warnings
import torch
import multiprocessing
import sentencepiece as spm
import time
from run import InputFeatures, convert_examples_to_features
from model import Model
from run import TextDataset
from utils import PLBARTCodeDataset
from utils import set_seed
from utils import Recorder_style
from attacker import Style_Attacker, get_code_pairs, get_transfered_code
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaModel, RobertaTokenizer)
from transformers import logging
from fairseq.models.bart import BARTModel

logging.set_verbosity_error()
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning
warnings.filterwarnings("ignore")

class Dictionary():
    def __init__(self, vocab_size, pad_id) -> None:
        self.vocab_size = vocab_size
        self.pad_id = pad_id

    def __len__(self):
        return self.vocab_size

    def pad(self):
        return self.pad_id
class Task():
    def __init__(self, vocab_size, pad_id) -> None:
        self.source_dictionary = Dictionary(vocab_size, pad_id)
        self.target_dictionary = self.source_dictionary
def build_mbart(args):
    task = Task(args.vocab_size, args.pad_id)
    model = BARTModel.build_model(args, task)
    return model

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--base_model", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--csv_store_path1", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--use_ga", action='store_true',
                        help="Whether to GA-Attack.")
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--beam_size', type=int, default=3,
                        help="beam_size")
    parser.add_argument('--start_num', type=int, default=0,
                        help="start_num")
    parser.add_argument('--end_num', type=int, default=3999,
                        help="end_num")

    args = parser.parse_args()

    device = torch.device("cuda")
    args.device = device

    # Set seed
    set_seed(args.seed)

    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer_name)
    config = argparse.Namespace(activation_fn='gelu', adam_betas='(0.9, 0.98)', adam_eps=1e-08, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, add_prev_output_tokens=True, all_gather_list_size=16384, arch='mbart_base', attention_dropout=0.1, batch_size=4, batch_size_valid=4, best_checkpoint_metric='accuracy', bf16=False, bpe=None, broadcast_buffers=False, bucket_cap_mb=25, checkpoint_shard_count=1, checkpoint_suffix='', classification_head_name='sentence_classification_head', clip_norm=1.0, cpu=False, criterion='sentence_prediction', cross_self_attention=False, curriculum=0, data='/home/zzr/CodeStudy/Defect-detection/plbart/processed/data-bin', data_buffer_size=10, dataset_impl=None, ddp_backend='no_c10d', decoder_attention_heads=12, decoder_embed_dim=768, decoder_embed_path=None, decoder_ffn_embed_dim=3072, decoder_input_dim=768, decoder_layerdrop=0, decoder_layers=6, decoder_layers_to_keep=None, decoder_learned_pos=True, decoder_normalize_before=False, decoder_output_dim=768, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=False, distributed_num_procs=0, distributed_port=-1, distributed_rank=0, distributed_world_size=1, distributed_wrapper='DDP', dropout=0.1, empty_cache_freq=0, encoder_attention_heads=12, encoder_embed_dim=768, encoder_embed_path=None, encoder_ffn_embed_dim=3072, encoder_layerdrop=0, encoder_layers=6, encoder_layers_to_keep=None, encoder_learned_pos=True, encoder_normalize_before=False, end_learning_rate=0.0, fast_stat_sync=False, find_unused_parameters=True, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=None, force_anneal=None, fp16=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test', init_token=0, keep_best_checkpoints=-1, keep_interval_updates=-1, keep_last_epochs=-1, langs='java,python,en_XX', layernorm_embedding=True, local_rank=0, localsgd_frequency=3, log_format='json', log_interval=10, lr=[5e-05], lr_scheduler='polynomial_decay', max_epoch=5, max_positions=512, max_source_positions=1024, max_target_positions=1024, max_tokens=2048, max_tokens_valid=2048, max_update=15000, maximize_best_checkpoint_metric=True, memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, min_lr=-1.0, model_parallel_size=1, no_cross_attention=False, no_epoch_checkpoints=True, no_last_checkpoints=False, no_progress_bar=False, no_save=False, no_save_optimizer_state=False, no_scale_embedding=False, no_seed_provided=False, no_shuffle=False, no_token_positional_embeddings=False, nprocs_per_node=1, num_classes=2, num_shards=1, num_workers=1, optimizer='adam', optimizer_overrides='{}', patience=-1, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, pooler_activation_fn='tanh', pooler_dropout=0.0, power=1.0, profile=False, quant_noise_pq=0, quant_noise_pq_block_size=8, quant_noise_scalar=0, quantization_config_path=None, regression_target=False, relu_dropout=0.0, required_batch_size_multiple=1, required_seq_len_multiple=1, reset_dataloader=True, reset_lr_scheduler=False, reset_meters=True, reset_optimizer=True, restore_file='/data2/cg/CodeStudy/PLBART/pretrain/checkpoint_11_100000.pt', save_dir='/home/zzr/CodeStudy/Defect-detection/plbart/devign', save_interval=1, save_interval_updates=0, scoring='bleu', seed=1234, sentence_avg=False, separator_token=None, shard_id=0, share_all_embeddings=True, share_decoder_input_output_embed=True, shorten_data_split_list='', shorten_method='truncate', skip_invalid_size_inputs_valid_test=False, slowmo_algorithm='LocalSGD', slowmo_momentum=None, stop_time_hours=0, task='plbart_sentence_prediction', tensorboard_logdir=None, threshold_loss_scale=None, tokenizer=None, total_num_update=1000000, tpu=False, train_subset='train', update_freq=[4], use_bmuf=False, use_old_adam=False, user_dir='/home/zzr/CodeStudy/PLBART/source', valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, warmup_updates=500, weight_decay=0.0, zero_sharding='none')
    config.num_labels = 1
    config.vocab_size = tokenizer.GetPieceSize() + 5
    config.pad_id = 1
    assert config.vocab_size == 50005

    max_len_single_sentence = 510
    if args.block_size <= 0:
        args.block_size = max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, max_len_single_sentence)

    model = build_mbart(config)
    if args.model_name_or_path:
        sd = torch.load(args.model_name_or_path, map_location="cpu")
        model.load_state_dict(sd["model"])

    model = Model(model, config, tokenizer, args)

    load_model_path = '../saved_models/checkpoint-best-f1/model.bin'
    model.load_state_dict(torch.load(load_model_path))
    print("{} - reload model from {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), load_model_path))

    model.to(args.device)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)

    ## Load Dataset
    cpu_cont = 16
    pool = multiprocessing.Pool(cpu_cont)
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file, pool=pool)
    ## Load code pairs
    source_codes = get_code_pairs(args.eval_data_file, tokenizer, args)
    print("131")
    subs_path = "../../../dataset/preprocess/test_subs_clone.jsonl"
    labels = []
    code1_list = []
    with open(subs_path) as f:
        for line in f:
            js = json.loads(line.strip())
            code1_list.append(js['code1'])
            labels.append(js["label"])

    recoder1 = Recorder_style(args.csv_store_path1)
    attacker = Style_Attacker(args, model, tokenizer, tokenizer_mlm)
    print("ATTACKER BUILT!")
    success_attack = 0
    total_cnt = 0
    start_time = time.time()
    index_start = 0
    index_end = 4000
    print("index_start:", index_start)
    print("index_end:", index_end)
    for index, example in enumerate(eval_dataset):
        if index < index_start:
            continue
        if index >= index_end:
            break
        if 1 == 1:
            print("Index: ", index)
            code_pair = source_codes[index]
            true_label = labels[index]
            orig_prob, orig_label = model.get_results([example], args.eval_batch_size)
            orig_prob = orig_prob[0]
            orig_label = orig_label[0]

            if orig_label != true_label:
                continue
            first_code = code_pair[2].strip()
            adv_codes = get_transfered_code(first_code)
            if len(adv_codes) == 0:
                print("len(adv_codes) == 0")
                # print("first_code", first_code)
                continue
            query_times = 0
            code_2 = code_pair[3]
            code_2 = " ".join(code_2.split())
            words_2 = tokenizer.tokenize(code_2)
            adv_codes_attack = []
            example_start_time = time.time()
            total_cnt += 1
            is_success = -1
            print("attack(1)_len(adv_codes)", len(adv_codes))
            for adv_code in adv_codes:
                new_adv_codes = get_transfered_code(adv_code)
                adv_codes_attack += new_adv_codes
                tmp_code = ' '.join(adv_code.split())
                tmp_feature = convert_examples_to_features(tmp_code, code_2, true_label, None, None,
                                                           tokenizer, args, None)
                new_dataset = PLBARTCodeDataset([tmp_feature])
                logits, preds = model.get_results(new_dataset, args.eval_batch_size)
                query_times += 1
                example_end_time = (time.time() - example_start_time) / 60
                if preds[0] != true_label:
                    is_success = 1
                    print("%s SUC! (%.5f => %.5f)" % \
                          ('>>', true_label, preds[0]), flush=True)
                    # return is_success, adv_code, query_times
                    print("Example time cost: ", round(example_end_time, 2), "min")
                    print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
                    print("Query times in this attack: ", query_times)
                    success_attack += 1
                    adv_label = "1" if true_label == 0 else "0"
                    recoder1.write(index, code_pair[0], adv_code, true_label,
                                  adv_label, query_times, round(example_end_time, 2), "style change")
                    # recoder3.write(index, code_pair[0].replace("\n", " "), adv_code.replace("\n", " "), true_label,
                    #                adv_label, query_times, round(example_end_time, 2))
                    print("Success rate is : {}/{} = {}".format(success_attack, total_cnt,
                                                                1.0 * success_attack / total_cnt))
                    break
                # else:
                #     adv_label = "1" if true_label == 0 else "0"
                #     recoder1.write(index, None, None, true_label,
                #                    adv_label, query_times, round(example_end_time, 2), "0")

            # fail
            if is_success == -1:
                print("Attack 1 failed on index = {}.".format(index))
                # t0 = time.time()
                while True:

                    is_success, adv_code, query_times_ = attacker.style_attack(code_pair, true_label, example, adv_codes_attack, query_times)
                    # print("Attack 3 time cost: ", round((time.time() - t0) / 60, 2), "min")
                    adv_codes_attack = adv_code
                    if query_times_ ==None:
                        query_times_ = 0
                    query_times += query_times_
                    if is_success == 1:
                        break
                    if query_times >= 500:
                        break
                # if not is_success == 1:
                #     print("Attack 3 failed on index = {}.".format(index))
                #     print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
                #     continue
                example_end_time = (time.time() - example_start_time) / 60
                if is_success == 1:

                    success_attack += 1
                    adv_label = "1" if true_label == 0 else "0"
                    recoder1.write(index, code_pair[0], adv_code, true_label,
                                  adv_label, query_times, round(example_end_time, 2), "style change")

                else:
                    adv_label = "1" if true_label == 0 else "0"
                    recoder1.write(index, None, None, true_label,
                                   adv_label, query_times, round(example_end_time, 2), "0")
                print("Example time cost: ", round(example_end_time, 2), "min")
                print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
                print("Query times in this attack: ", query_times)
                print(
                    "Success rate is : {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))




if __name__ == '__main__':
    main()
