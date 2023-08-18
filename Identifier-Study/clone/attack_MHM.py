# coding=gbk
import json
import sys
import os

sys.path.append('../../CodeBERT/Clone-detection/code')
sys.path.append('../../')
sys.path.append('../../python_parser')
retval = os.getcwd()

import csv
import logging
import argparse
import warnings
import pickle
import pandas as pd
from tqdm import tqdm
import copy
import torch
import time
import numpy as np

from model import Model
import multiprocessing
from utils import set_seed
from utils import Recorder
from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)

import copy
import torch
from attacker import MHM_Attacker
import logging
import random
import utils
from utils import select_parents, crossover, map_chromesome, mutate, is_valid_variable_name, _tokenize, \
    get_identifier_posistions_from_code, get_masked_code_by_position, get_substitues, is_valid_substitue, set_seed
from python_parser.run_parser import get_identifiers, remove_comments_and_docstrings, get_example, get_example_batch
from utils import is_valid_variable_name, _tokenize, get_identifier_posistions_from_code, get_masked_code_by_position, \
    get_substitues, is_valid_substitue
from attacker import TextDataset, convert_examples_to_features
from utils import CodeDataset
from utils import getUID, isUID, getTensor, build_vocab
from run_parser import get_identifiers, get_example
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

logger = logging.getLogger(__name__)


def get_code_pairs(file_path, tokenizer, args):
    url_to_code = {}
    with open('/'.join(file_path.split('/')[:-1]) + '/data.jsonl') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            url_to_code[js['idx']] = js['func']
    data = []
    cache = {}
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data.append((url1, url2, label, tokenizer, args, cache, url_to_code))
    code_pairs = []
    for sing_example in data:
        code_pairs.append([sing_example[0],
                           sing_example[1],
                           url_to_code[sing_example[0]],
                           url_to_code[sing_example[1]]])
    return code_pairs
MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)
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

    args = parser.parse_args()
    device = torch.device("cuda")
    args.device = device

    set_seed(args.seed)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path, cache_dir=None)
    config.num_labels = 2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=False, cache_dir=None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config, cache_dir=None)

    model = Model(model, config, tokenizer, args)
    checkpoint_prefix = 'checkpoint-best-f1/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)

    ## Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    codebert_mlm.to('cuda')
    cpu_cont = 16
    ## Load tensor features
    eval_data_file = "data/{}_statement.txt".format(args.statement)
    pool = multiprocessing.Pool(cpu_cont)
    eval_dataset = TextDataset(tokenizer, args, eval_data_file, pool=pool)

    ## Load code pairs
    source_codes = get_code_pairs(eval_data_file, tokenizer, args)
    eval_list = []
    with open(eval_data_file) as f:
        for line in f:
            line = line.strip()
            url1, url2, label = line.split('\t')
            eval_list.append(url1)
    print(len(eval_list))

    file = "data/data.csv"
    df = pd.read_csv(file, encoding="utf-8")
    # print(df.columns)
    code_list = df["code"].tolist()
    idx_list = df["idx"].tolist()
    state_list = df["{}_statement".format(args.statement)].tolist()

    subs_path = "data/data_subs_test.jsonl"
    substitutes_list = {}
    with open(subs_path) as f:
        for line in f:
            js = json.loads(line.strip())
            substitutes_list[js["idx"]] = js["substitutes"]

    substitutes = []
    for idx in eval_list:
        substitutes.append(substitutes_list[idx])

    assert len(source_codes) == len(eval_dataset) == len(substitutes)

    code_tokens = []
    for index, code in enumerate(source_codes):
        code_tokens.append(get_identifiers(code[2], "java")[1])

    id2token, token2id = build_vocab(code_tokens, 5000)

    attacker = MHM_Attacker(args, model, codebert_mlm, tokenizer_mlm, token2id, id2token)

    attack_succ_1 = 0
    attack_succ_3 = 0
    recoder_1 = Recorder("result/attack_mhm_" + args.statement + "_1.csv")
    recoder_3 = Recorder("result/attack_mhm_" + args.statement + "_3.csv")
    total_cnt = 0
    query_times = 0
    for index, example in enumerate(tqdm(eval_dataset)):
        print("Index: ", index)
        substitute = substitutes[index]
        substitute_key = list(substitute.keys())
        print("substitute:", substitute_key)
        code_pair = source_codes[index]
        code_1 = code_pair[2]
        code_2 = code_pair[3]
        logits, preds = model.get_results([example], args.eval_batch_size)
        orig_label = preds[0]
        orig_prob = logits[0][orig_label]
        # current_prob = max(orig_prob)

        true_label = example[1].item()
        if not true_label == orig_label:
            continue

        code_2 = " ".join(code_2.split())
        words_2 = tokenizer_mlm.tokenize(code_2)

        idx = int(code_pair[0])

        statements = []
        for index in range(len(idx_list)):
            if idx_list[index] == idx:
                statements = state_list[index].replace(" ", "").strip('[').strip(']').split(',')
                statements = [] if statements == [''] else statements
        statements = [state for state in statements if state in substitute_key]
        print("statements:", statements)
        if len(statements) == 0:
            continue

        total_cnt += 1
        if len(statements) == 1:
            start_time = time.time()
            _res = attacker.mcmc_random(example, statements, substitute, tokenizer, code_pair,
                                        _label=true_label, _n_candi=30,
                                        _max_iter=10, _prob_threshold=1)
            if _res['succ'] == True:
                attack_succ_1 += 1
                attack_succ_3 += 1
                time_cost = (time.time() - start_time) / 60
                recoder_1.write(index,
                              code_1.replace("\n", " "),
                              _res['tokens'].replace("\n", " "), _res["old_uid"], _res["replace_info"],
                              _res["prog_length"],
                              len(substitute_key),
                              _res["nb_changed_var"],
                              _res["nb_changed_pos"],
                              model.query - query_times,
                              time_cost)
                recoder_3.write(index,
                                code_1.replace("\n", " "),
                                _res['tokens'].replace("\n", " "), _res["old_uid"], _res["replace_info"],
                                _res["prog_length"],
                                len(substitute_key),
                                _res["nb_changed_var"],
                                _res["nb_changed_pos"],
                                model.query - query_times,
                                time_cost)
        elif len(statements) <= 3:
            flag = 0
            statements_1 = random.sample(statements, 1)
            start_time = time.time()
            _res_1 = attacker.mcmc_random(example, statements_1, substitute, tokenizer, code_pair,
                                          _label=true_label, _n_candi=30,
                                          _max_iter=10, _prob_threshold=1)
            if _res_1['succ'] == True:
                attack_succ_1 += 1
                attack_succ_3 += 1
                time_cost = (time.time() - start_time) / 60
                recoder_1.write(index,
                                code_1.replace("\n", " "),
                                _res_1['tokens'].replace("\n", " "), _res_1["old_uid"], _res_1["replace_info"],
                                _res_1["prog_length"],
                                len(substitute_key),
                                _res_1["nb_changed_var"],
                                _res_1["nb_changed_pos"],
                                model.query - query_times,
                                time_cost)
                recoder_3.write(index,
                                code_1.replace("\n", " "),
                                _res_1['tokens'].replace("\n", " "), _res_1["old_uid"], _res_1["replace_info"],
                                _res_1["prog_length"],
                                len(substitute_key),
                                _res_1["nb_changed_var"],
                                _res_1["nb_changed_pos"],
                                model.query - query_times,
                                time_cost)
                flag = 1
            if flag == 0:
                start_time = time.time()
                _res = attacker.mcmc_random(example, statements, substitute, tokenizer, code_pair,
                                            _label=true_label, _n_candi=30,
                                            _max_iter=10, _prob_threshold=1)
                if _res['succ'] == True:
                    attack_succ_3 += 1
                    time_cost = (time.time() - start_time) / 60
                    recoder_3.write(index,
                                    code_1.replace("\n", " "),
                                    _res['tokens'].replace("\n", " "), _res["old_uid"], _res["replace_info"],
                                    _res["prog_length"],
                                    len(substitute_key),
                                    _res["nb_changed_var"],
                                    _res["nb_changed_pos"],
                                    model.query - query_times,
                                    time_cost)

        elif len(statements) > 3:
            flag = 0
            statements_1 = random.sample(statements, 1)
            start_time = time.time()
            _res_1 = attacker.mcmc_random(example, statements_1, substitute, tokenizer, code_pair,
                                          _label=true_label, _n_candi=30,
                                          _max_iter=10, _prob_threshold=1)
            if _res_1['succ'] == True:
                attack_succ_1 += 1
                attack_succ_3 += 1
                time_cost = (time.time() - start_time) / 60
                recoder_1.write(index,
                                code_1.replace("\n", " "),
                                _res_1['tokens'].replace("\n", " "), _res_1["old_uid"], _res_1["replace_info"],
                                _res_1["prog_length"],
                                len(substitute_key),
                                _res_1["nb_changed_var"],
                                _res_1["nb_changed_pos"],
                                model.query - query_times,
                                time_cost)
                recoder_3.write(index,
                                code_1.replace("\n", " "),
                                _res_1['tokens'].replace("\n", " "), _res_1["old_uid"], _res_1["replace_info"],
                                _res_1["prog_length"],
                                len(substitute_key),
                                _res_1["nb_changed_var"],
                                _res_1["nb_changed_pos"],
                                model.query - query_times,
                                time_cost)
                flag = 1
            if flag == 0:
                statements_3 = random.sample(statements, 3)
                start_time = time.time()
                _res_3 = attacker.mcmc_random(example, statements_3, substitute, tokenizer, code_pair,
                                        _label=true_label, _n_candi=30,
                                        _max_iter=10, _prob_threshold=1)
                if _res_3['succ'] == True:
                    attack_succ_3 += 1
                    time_cost = (time.time() - start_time) / 60
                    recoder_3.write(index,
                                    code_1.replace("\n", " "),
                                    _res_3['tokens'].replace("\n", " "), _res_3["old_uid"], _res_3["replace_info"],
                                    _res_3["prog_length"],
                                    len(substitute_key),
                                    _res_3["nb_changed_var"],
                                    _res_3["nb_changed_pos"],
                                    model.query - query_times,
                                    time_cost)

        print("\nattack_succ with 1: {}/{}={}".format(attack_succ_1, total_cnt, (attack_succ_1 / total_cnt)))
        print("attack_succ with 3: {}/{}={}".format(attack_succ_3, total_cnt, (attack_succ_3 / total_cnt)))

    print("Final attack_succ with 1: {}/{}={}".format(attack_succ_1, total_cnt, (attack_succ_1 / total_cnt)))
    print("Final attack_succ with 3: {}/{}={}".format(attack_succ_3, total_cnt, (attack_succ_3 / total_cnt)))

if __name__ == '__main__':
    main()
