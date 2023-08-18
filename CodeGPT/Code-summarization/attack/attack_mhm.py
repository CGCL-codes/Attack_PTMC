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
import random
from run_parser import get_identifiers, get_example
from utils import set_seed
from model import Seq2Seq
from utils import Recorder, isUID, get_code_tokens
from run import TextDataset, convert_examples_to_features
from attacker import MHM_Attacker, read_examples
from transformers import RobertaForMaskedLM
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from utils import select_parents, crossover, map_chromesome, mutate, is_valid_variable_name, _tokenize, \
    get_identifier_posistions_from_code, get_masked_code_by_position, get_substitues, is_valid_substitue, set_seed


warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
                 }

logger = logging.getLogger(__name__)

from utils import build_vocab

def update_config(model, tokenizer):
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

if __name__ == "__main__":
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
    parser.add_argument("--csv_store_path", default=None, type=str,
                        help="Base Model")
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
    # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
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
    model.to(device)

    ## Load code
    eval_examples = read_examples(args.dev_filename)

    subs_path = "../../../dataset/preprocess/test_subs_summary_sampled.jsonl"
    substitutes = []
    with open(subs_path) as f:
        for line in f:
            js = json.loads(line.strip())
            substitutes.append(js["substitutes"])
    assert len(eval_examples) == len(substitutes)

    code_tokens = []
    for index, example in enumerate(eval_examples):
        source_code = example.source
        code_tokens.append(get_identifiers(source_code, "java")[1])

    id2token, token2id = build_vocab(code_tokens, 5000)
    bleu_file = "mhm"
    if os.path.exists(bleu_file) is False:
        os.makedirs(bleu_file)
    recoder = Recorder(args.csv_store_path)
    attacker = MHM_Attacker(args, model, tokenizer, token2id, id2token, bleu_file)
    success_attack = 0
    total_cnt = 0

    print("ATTACKER BUILT!")
    start_time = time.time()
    query_times = 0
    for index, example in enumerate(eval_examples):
        print("Index: ", index)
        original_bleu, pre_summary, ref_summary = attacker.eval_bleu(example)
        if original_bleu == 0:
            continue
        code = example.source
        orig_code_tokens = get_code_tokens(code)
        identifiers, code_tokens = get_identifiers(code, 'java')
        prog_length = len(code_tokens)
        substitute = substitutes[index]
        if len(identifiers) == 0:
            continue
        total_cnt += 1

        # Start MHM attack
        example_start_time = time.time()
        result = attacker.mcmc_random(tokenizer, example, substitute, code,
                                      _label=ref_summary, _n_candi=30,
                                      _max_iter=100, _prob_threshold=1)

        example_end_time = (time.time() - example_start_time) / 60
        replace_info = ''
        if result["replace_info"] is not None:
            for key in result["replace_info"].keys():
                replace_info += key + ':' + result["replace_info"][key] + ','

        print("Example time cost: ", round(example_end_time, 2), "min")
        print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
        print("Query times in this attack: ", attacker.query - query_times)
        if result["is_success"] == 1:
            success_attack += 1
            recoder.write(index, code, result["tokens"], len(orig_code_tokens), len(identifiers), replace_info,
                          attacker.query - query_times, round(example_end_time, 2), "ALERT")
        else:
            recoder.write(index, None, None,
                          len(orig_code_tokens), len(identifiers),
                          None, attacker.query - query_times, example_end_time, "0")
        query_times = attacker.query
        print("Success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))
    print("Finall success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))


