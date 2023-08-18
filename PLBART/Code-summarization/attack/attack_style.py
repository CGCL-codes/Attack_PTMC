# coding=utf-8
# @Time    : 2020/7/8
# @Author  : Zhou Yang
# @Email   : zyang@smu.edu.sg
# @File    : attack.py
'''For attacking CodeBERT models'''
import json
import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('../../../')
sys.path.append('../code')
sys.path.append('../../../python_parser')
retval = os.getcwd()

import csv
import logging
import argparse
import warnings
import pickle
import copy
import sentencepiece as spm
import torch
import torch.nn as nn
import multiprocessing
import time
import numpy as np
from utils import set_seed
from model import Seq2Seq
from utils import Recorder_summary_style
from run import TextDataset, convert_examples_to_features, build_mbart
from attacker import Style_Attacker, get_transfered_code,get_new_example
from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset


warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            else:
                idx = js['idx']
            code = js['code']
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                        idx=idx,
                        source=code,
                        target=nl,
                        )
            )
            # if len(examples) > 1000: break
    return examples

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--base_model", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--use_ga", action='store_true',
                        help="Whether to GA-Attack.")
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
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
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--csv_store_path1", default=None, type=str,
                        help="Base Model")
    args = parser.parse_args()

    device = torch.device("cuda")
    args.device = device

    # Set seed
    set_seed(args.seed)

    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer_name)
    config = argparse.Namespace(activation_fn='gelu', adam_betas='(0.9, 0.98)', adam_eps=1e-08, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, add_prev_output_tokens=True, all_gather_list_size=16384, arch='mbart_base', attention_dropout=0.1, batch_size=4, batch_size_valid=4, best_checkpoint_metric='accuracy', bf16=False, bpe=None, broadcast_buffers=False, bucket_cap_mb=25, checkpoint_shard_count=1, checkpoint_suffix='', classification_head_name='sentence_classification_head', clip_norm=1.0, cpu=False, criterion='sentence_prediction', cross_self_attention=False, data_buffer_size=10, dataset_impl=None, ddp_backend='no_c10d', decoder_attention_heads=12, decoder_embed_dim=768, decoder_embed_path=None, decoder_ffn_embed_dim=3072, decoder_input_dim=768, decoder_layerdrop=0, decoder_layers=6, decoder_layers_to_keep=None, decoder_learned_pos=True, decoder_normalize_before=False, decoder_output_dim=768, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=False, distributed_num_procs=0, distributed_port=-1, distributed_rank=0, distributed_world_size=1, distributed_wrapper='DDP', dropout=0.1, empty_cache_freq=0, encoder_attention_heads=12, encoder_embed_dim=768, encoder_embed_path=None, encoder_ffn_embed_dim=3072, encoder_layerdrop=0, encoder_layers=6, encoder_layers_to_keep=None, encoder_learned_pos=True, encoder_normalize_before=False, end_learning_rate=0.0, fast_stat_sync=False, find_unused_parameters=True, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=None, force_anneal=None, fp16=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test', init_token=0, keep_best_checkpoints=-1, keep_interval_updates=-1, keep_last_epochs=-1, langs='java,python,en_XX', layernorm_embedding=True, local_rank=0, localsgd_frequency=3, log_format='json', log_interval=10, lr=[5e-05], lr_scheduler='polynomial_decay', max_epoch=5, max_positions=512, max_source_positions=1024, max_target_positions=1024, max_tokens=2048, max_tokens_valid=2048, max_update=15000, maximize_best_checkpoint_metric=True, memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, min_lr=-1.0, model_parallel_size=1, no_cross_attention=False, no_epoch_checkpoints=True, no_last_checkpoints=False, no_progress_bar=False, no_save=False, no_save_optimizer_state=False, no_scale_embedding=False, no_seed_provided=False, no_shuffle=False, no_token_positional_embeddings=False, nprocs_per_node=1, num_classes=2, num_shards=1, num_workers=1, optimizer='adam', optimizer_overrides='{}', patience=-1, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, pooler_activation_fn='tanh', pooler_dropout=0.0, power=1.0, profile=False, quant_noise_pq=0, quant_noise_pq_block_size=8, quant_noise_scalar=0, quantization_config_path=None, regression_target=False, relu_dropout=0.0, required_batch_size_multiple=1, required_seq_len_multiple=1, reset_dataloader=True, reset_lr_scheduler=False, reset_meters=True, reset_optimizer=True, save_interval=1, save_interval_updates=0, scoring='bleu', seed=1234, sentence_avg=False, separator_token=None, shard_id=0, share_all_embeddings=True, share_decoder_input_output_embed=True, shorten_data_split_list='', shorten_method='truncate', skip_invalid_size_inputs_valid_test=False, slowmo_algorithm='LocalSGD', slowmo_momentum=None, stop_time_hours=0, task='plbart_sentence_prediction', tensorboard_logdir=None, threshold_loss_scale=None, tokenizer=None, total_num_update=1000000, tpu=False, train_subset='train', update_freq=[4], use_bmuf=False, use_old_adam=False, valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, warmup_updates=500, weight_decay=0.0, zero_sharding='none')
    config.num_labels = 1
    config.vocab_size = tokenizer.GetPieceSize() + 5
    config.pad_id = 1
    assert config.vocab_size == 50005

    # budild model
    model = build_mbart(config)
    if args.model_name_or_path:
        sd = torch.load(args.model_name_or_path, map_location="cpu")
        model.load_state_dict(sd["model"])
        logger.info("Reload model from {}.".format(args.model_name_or_path))

    model = Seq2Seq(encoder=model.encoder, decoder=model.decoder, config=config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=0, eos_id=2)
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        logger.info("model loaded successfully.")

    model.to(device)

    ## Load code
    eval_examples = read_examples(args.dev_filename)

    bleu_file = "style"
    if os.path.exists(bleu_file) is False:
        os.makedirs(bleu_file)

    recoder1 = Recorder_summary_style(args.csv_store_path1)
    # recoder3 = Recorder_summary_style(args.csv_store_path2)
    attacker = Style_Attacker(args, model, tokenizer,bleu_file)

    success_attack = 0
    total_cnt = 0
    start_time = time.time()
    print("ATTACKER BUILT!")
    index_start = 0
    index_end = 4000
    print("index_start:", index_start)
    print("index_end:", index_end)
    for index, example in enumerate(eval_examples):
        if index < index_start:
            continue
        if index >= index_end:
            break
        if 1 == 1:
            print("Index: ", index)
            original_bleu, pre_summary, ref_summary = attacker.eval_bleu(example, model, tokenizer)
            if original_bleu == 0:
                continue
            code = example.source
            adv_codes = get_transfered_code(code)
            if len(adv_codes) == 0:
                continue
            total_cnt += 1

            example_start_time = time.time()

            # attack_num: 1
            query_times = 0
            is_success = -1
            adv_codes_attack = []
            print("attack(1)_len(adv_codes)", len(adv_codes))
            for adv_code in adv_codes:
                new_adv_codes = get_transfered_code(adv_code)
                adv_codes_attack += new_adv_codes
                new_example = get_new_example(example.idx, adv_code, example.target)
                pred = attacker.eval_bleu(new_example[0], model, tokenizer)
                query_times += 1
                example_end_time = (time.time() - example_start_time) / 60
                if pred[0] == 0.0:
                    is_success = 1
                    print("%s SUC! (%.5f => %.5f)" % \
                          ('>>', original_bleu, pred[0]), flush=True)
                    # return is_success, new_adv_codes, pred[1], query_times
                    print("Attack_1 SUC on index = {}.".format(index))

                    print("Example time cost: ", round(example_end_time, 2), "min")
                    print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
                    print("Query times in this attack: ", query_times)
                    success_attack += 1

                    recoder1.write(index, code, adv_code, ref_summary, pre_summary,
                                   pred[1],
                                   query_times, round(example_end_time, 2), "style change")
                    print(
                        "Success rate is : {}/{} = {}".format(success_attack, total_cnt,
                                                              1.0 * success_attack / total_cnt))
                    break
            if is_success == -1:
                print("Attack_1 failed on index = {}.".format(index))
                while True:

                    is_success, adv_code, adv_summary, query_times_ = attacker.style_attack(original_bleu, example,
                                                                                            new_adv_codes, query_times)
                    new_adv_codes = adv_code
                    query_times += query_times_
                    if is_success == 1:
                        break
                    if query_times >= 500:
                        break
                example_end_time = (time.time() - example_start_time) / 60
                if is_success == 1:
                    print("Example time cost: ", round(example_end_time, 2), "min")
                    print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
                    print("Query times in this attack: ", query_times)
                    success_attack += 1
                    recoder1.write(index, code, adv_code, ref_summary,
                                   pre_summary,
                                   adv_summary,
                                   query_times, round(example_end_time, 2), "style change")
                    print(
                        "Success rate is : {}/{} = {}".format(success_attack, total_cnt,
                                                              1.0 * success_attack / total_cnt))
                else:
                    recoder1.write(index, None, None, None,
                                   None,
                                   None,
                                   query_times, round(example_end_time, 2), "0")
                    print("Example time cost: ", round(example_end_time, 2), "min")
                    print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
                    print("Query times in this attack: ", query_times)
                    print(
                        "Success rate is : {}/{} = {}".format(success_attack, total_cnt,
                                                              1.0 * success_attack / total_cnt))
            print("Success rate is : {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))


if __name__ == '__main__':
    main()