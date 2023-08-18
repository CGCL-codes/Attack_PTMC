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
import torch
import torch.nn as nn
import multiprocessing
import time
import numpy as np
from utils import set_seed
from model import Seq2Seq
from utils import Recorder_summary_style
from run import TextDataset, convert_examples_to_features, read_examples
from attacker import Style_Attacker, get_transfered_code, get_new_example
from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset


warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
                 }

logger = logging.getLogger(__name__)

def update_config(model, tokenizer):
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

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

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case,
                                                bos_token='<s>', eos_token='</s>', pad_token='<pad>',
                                                unk_token='<|UNKNOWN|>', sep_token='concode_elem_sep')

    # budild model
    decoder = model_class.from_pretrained(args.model_name_or_path)
    decoder.resize_token_embeddings(len(tokenizer))
    update_config(decoder, tokenizer)
    model = Seq2Seq(decoder=decoder, config=decoder.config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.bos_token_id, eos_id=tokenizer.eos_token_id)
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
    attacker = Style_Attacker(args, model, tokenizer, bleu_file)

    success_attack = 0
    total_cnt = 0
    start_time = time.time()
    print("ATTACKER BUILT!")
    logger.info("ATTACKER BUILT!")
    index_start = 0
    index_end = 4000
    print("index_start:", index_start)
    print("index_end:", index_end)
    for index, example in enumerate(eval_examples):
        if index < index_start:
            continue
        if index >= index_end:
            break
        print("Index: ", index)
        # logger.info("Index: ", index)
        original_bleu, pre_summary, ref_summary = attacker.eval_bleu(example, model, tokenizer)
        if original_bleu == 0:
            continue
        code = example.source
        adv_codes = get_transfered_code(code)
        if len(adv_codes) == 0:
            continue
        total_cnt += 1
        print("Pre summary: ", pre_summary)
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
                    "Success rate is : {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))
                break
        if is_success == -1:
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