# coding=gbk
import json
import sys
import os

sys.path.append('../../')
sys.path.append('../../python_parser')
retval = os.getcwd()

import csv
import logging
import argparse
import warnings
import pickle
import pandas as pd
import copy
import torch
import time
import numpy as np

from model import Model
import multiprocessing
from utils import set_seed, is_valid_identifier, get_code_tokens
from utils import Recorder
from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)

import copy
import torch
from datetime import datetime
from attacker import MHM_Attacker
import random
import utils
from utils import select_parents, crossover, map_chromesome, mutate, is_valid_variable_name, _tokenize, \
    get_identifier_posistions_from_code, get_masked_code_by_position, get_substitues, is_valid_substitue, set_seed
from python_parser.run_parser import get_identifiers, remove_comments_and_docstrings, get_example, get_example_batch
from utils import is_valid_variable_name, _tokenize, get_identifier_posistions_from_code, get_masked_code_by_position, \
    get_substitues, is_valid_substitue
from attacker import TextDataset, convert_examples_to_features, convert_code_to_features
from utils import CodeDataset
from utils import getUID, isUID, getTensor, build_vocab
from run_parser import get_identifiers, get_example
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import logging
logging.set_verbosity_error()
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning
warnings.filterwarnings("ignore")

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}


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
    parser.add_argument('--statement', type=str, default="",
                        help="type of identifier statement")

    def filter_state(code, statements):
        code_token = get_code_tokens(code)
        filter_identifier = []
        for identifier in statements:
            if is_valid_identifier(identifier):
                position = []
                for index, token in enumerate(code_token):
                    if identifier == token:
                        position.append(index)
                if not all(x > args.block_size - 2 for x in position):
                    filter_identifier.append(identifier)
        return filter_identifier

    args = parser.parse_args()
    device = torch.device("cuda")
    args.device = device

    set_seed(args.seed)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path, cache_dir=None)
    config.num_labels = 1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=False, cache_dir=None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config, cache_dir=None)

    model = Model(model, config, tokenizer, args)
    checkpoint_prefix = 'checkpoint-best-acc/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)
    print("{} - reload model from {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), output_dir))

    ## Load tensor features
    eval_data_file = "data/{}_statement.csv".format(args.statement)
    eval_dataset = TextDataset(tokenizer, args, eval_data_file)

    df = pd.read_csv(eval_data_file, encoding="utf-8")
    # print(df.columns)
    code_list = df["code"].tolist()
    idx_list = df["idx"].tolist()
    state_list = df["identifier"]

    subs_path = "data/data_subs_test.jsonl"
    substitutes_list = {}
    source_codes = []
    with open(subs_path) as f:
        for line in f:
            js = json.loads(line.strip())
            source_codes.append(js['func'])
            substitutes_list[js["idx"]] = js["substitutes"]

    substitutes = []
    for idx in idx_list:
        substitutes.append(substitutes_list[idx])

    assert len(code_list) == len(eval_dataset) == len(substitutes)

    code_tokens = []
    for index, code in enumerate(source_codes):
        code_tokens.append(get_identifiers(code, "java")[1])

    id2token, token2id = build_vocab(code_tokens, 5000)

    attacker = MHM_Attacker(args, model, tokenizer, token2id, id2token)

    attack_succ_1 = 0
    attack_succ_3 = 0

    total_cnt = 0
    query_times = 0
    for index, example in enumerate(eval_dataset):
        print("Index: ", index)
        substitute = substitutes[index]
        substitute_key = list(substitute.keys())
        code = code_list[index]
        logits, preds = model.get_results([example], args.eval_batch_size)
        orig_label = preds[0]
        orig_prob = logits[0][orig_label]
        # current_prob = max(orig_prob)

        true_label = example[1].item()
        if not true_label == orig_label:
            continue

        statements = []
        statements = state_list[index].replace(" ", "").replace("'", "").strip('[').strip(']').split(',')
        statements = [] if statements == [''] else statements
        statements = [state for state in statements if state in substitute_key]
        statements = filter_state(code, statements)
        print("statements:", statements)
        if len(statements) == 0:
            continue

        total_cnt += 1
        if len(statements) == 1:
            _res = attacker.mcmc_random(statements, substitute, tokenizer, code,
                                        _label=true_label, _n_candi=30,
                                        _max_iter=10, _prob_threshold=1)
            if _res['succ'] == True:
                attack_succ_1 += 1
                attack_succ_3 += 1

        elif len(statements) <= 3:
            flag = 0
            statements_1 = random.sample(statements, 1)
            _res_1 = attacker.mcmc_random(statements_1, substitute, tokenizer, code,
                                          _label=true_label, _n_candi=30,
                                          _max_iter=10, _prob_threshold=1)
            if _res_1['succ'] == True:
                attack_succ_1 += 1
                attack_succ_3 += 1
                flag = 1
            if flag == 0:
                _res = attacker.mcmc_random(statements, substitute, tokenizer, code,
                                            _label=true_label, _n_candi=30,
                                            _max_iter=10, _prob_threshold=1)
                if _res['succ'] == True:
                    attack_succ_3 += 1

        elif len(statements) > 3:
            flag = 0
            statements_1 = random.sample(statements, 1)
            _res_1 = attacker.mcmc_random(statements_1, substitute, tokenizer, code,
                                          _label=true_label, _n_candi=30,
                                          _max_iter=10, _prob_threshold=1)
            if _res_1['succ'] == True:
                attack_succ_1 += 1
                attack_succ_3 += 1

                flag = 1
            if flag == 0:
                statements_3 = random.sample(statements, 3)
                _res_3 = attacker.mcmc_random(statements_3, substitute, tokenizer, code,
                                        _label=true_label, _n_candi=30,
                                        _max_iter=10, _prob_threshold=1)
                if _res_3['succ'] == True:
                    attack_succ_3 += 1

        print("attack_succ with 1: {}/{}={}".format(attack_succ_1, total_cnt, (attack_succ_1 / total_cnt)))
        print("attack_succ with 3: {}/{}={}".format(attack_succ_3, total_cnt, (attack_succ_3 / total_cnt)))

    print("Final attack_succ with 1: {}/{}={}".format(attack_succ_1, total_cnt, (attack_succ_1 / total_cnt)))
    print("Final attack_succ with 3: {}/{}={}".format(attack_succ_3, total_cnt, (attack_succ_3 / total_cnt)))

if __name__ == '__main__':
    main()
