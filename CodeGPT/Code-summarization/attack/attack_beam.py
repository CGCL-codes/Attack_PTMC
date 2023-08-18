# coding=utf-8
import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('../../../')
sys.path.append('../code')
sys.path.append('../../../python_parser')
import json
import sys
from datetime import datetime
import torch.nn as nn
from model import Seq2Seq

sys.path.append('../../../')
sys.path.append('../code')
sys.path.append('../../../python_parser')
retval = os.getcwd()

import csv
import argparse
import warnings
import pickle
from tqdm import tqdm
import copy
import torch
import multiprocessing
import time
import numpy as np
from run_parser import get_identifiers, get_example
from utils import set_seed, get_code_tokens
from utils import Recorder
from attacker import read_examples
from beamAttacker import Beam_Attacker, get_statement_identifier
from transformers import RobertaForMaskedLM
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
from transformers import logging
logging.set_verbosity_error()
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning
warnings.filterwarnings("ignore")

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
                 }

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
    parser.add_argument("--csv_store_path", default=None, type=str,
                        help="Base Model")
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
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--beam_size_num', type=int, default=3,
                        help="beam_size")
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
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case, bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>', sep_token='concode_elem_sep')

    # budild model
    decoder = model_class.from_pretrained(args.model_name_or_path)
    decoder.resize_token_embeddings(len(tokenizer))
    update_config(decoder, tokenizer)
    model = Seq2Seq(decoder=decoder, config=decoder.config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.bos_token_id, eos_id=tokenizer.eos_token_id)
    if args.load_model_path is not None:
        model.load_state_dict(torch.load(args.load_model_path))
    print("{} - reload model from {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.load_model_path))

    model.to(device)

    codebert_mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    tokenizer_mlm = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    codebert_mlm.to('cuda')
    # Load Dataset
    ## Load Dataset
    eval_dataset = read_examples(args.eval_data_file)
    subs_path = "../../../Identifier-Study/summary/data/new_data.jsonl"
    generated_substitutions = []
    with open(subs_path) as f:
        for line in f:
            js = json.loads(line.strip())
            generated_substitutions.append(js["substitutes"])

    assert len(eval_dataset) == len(generated_substitutions)
    bleu_file = "beam"
    if os.path.exists(bleu_file) is False:
        os.makedirs(bleu_file)

    success_attack = 0
    total_cnt = 0
    recoder = Recorder(args.csv_store_path)
    attacker = Beam_Attacker(args, model, tokenizer, tokenizer_mlm, codebert_mlm, bleu_file)
    start_time = time.time()
    query_times = 0
    for index, example in enumerate(eval_dataset):
        print("Index: ", index)
        original_bleu, pre_summary, ref_summary = attacker.eval_bleu(example)
        if original_bleu == 0:
            continue
        code = example.source
        orig_code_tokens = get_code_tokens(code)
        substitutes = generated_substitutions[index]
        identifiers = list(substitutes.keys())
        if len(identifiers) == 0:
            continue
        total_cnt += 1
        statement_dict = get_statement_identifier(example.idx, identifiers)
        example_start_time = time.time()
        print(statement_dict)

        result = attacker.beam_attack(original_bleu, example, substitutes, statement_dict, args.beam_size_num)
        example_end_time = (time.time() - example_start_time) / 60
        print("Example time cost: ", round(example_end_time, 2), "min")
        print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
        print("Query times in this attack: ", attacker.query - query_times)
        if result["succ"] == 1:
            success_attack += 1
            recoder.write(index, code, result["adv_code"], len(orig_code_tokens), len(identifiers),
                          result["replace_info"], attacker.query - query_times, example_end_time, result["type"])
        else:
            recoder.write(index, None, None,
                          len(orig_code_tokens), len(identifiers),
                          None, attacker.query - query_times, example_end_time, "0")

        query_times = attacker.query
        print("Success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))
    print("Finall success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))


if __name__ == '__main__':
    main()