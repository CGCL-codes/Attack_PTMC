# coding=gbk

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import sys
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

import multiprocessing
from utils import set_seed, get_code_tokens, is_valid_identifier
from utils import Recorder
from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)

import copy
import torch
from attacker import MHM_Attacker, eval_bleu
import logging
import random
import utils
from utils import select_parents, crossover, map_chromesome, mutate, is_valid_variable_name, _tokenize, \
    get_identifier_posistions_from_code, get_masked_code_by_position, get_substitues, is_valid_substitue, set_seed
from python_parser.run_parser import get_identifiers, remove_comments_and_docstrings, get_example, get_example_batch
from utils import is_valid_variable_name, _tokenize, get_identifier_posistions_from_code, get_masked_code_by_position, \
    get_substitues, is_valid_substitue
from model import Seq2Seq
from utils import CodeDataset
from utils import Recorder_summary
import torch.nn as nn
from utils import getUID, isUID, getTensor, build_vocab
from run_parser import get_identifiers, get_example
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)
}

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
    df = pd.read_csv(filename, encoding="utf-8")
    code_list = df["code"].tolist()
    idx_list = df["idx"].tolist()
    summary_list = df["summary"].tolist()
    for i in range(len(idx_list)):
        idx = idx_list[i]
        code = code_list[i]
        nl = summary_list[i]
        examples.append(
            Example(
                idx=idx,
                source=code,
                target=nl,
            )
        )
    return examples

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
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--statement', type=str, default="",
                        help="type of identifier statement")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    def filter_state(code, statements):
        code_token = get_code_tokens(code)
        filter_identifier = []
        for identifier in statements:
            if is_valid_identifier(identifier):
                position = []
                for index, token in enumerate(code_token):
                    if identifier == token:
                        position.append(index)
                if not all(x > args.max_source_length - 2 for x in position):
                    filter_identifier.append(identifier)
        return filter_identifier

    args = parser.parse_args()
    device = torch.device("cuda")
    args.device = device

    set_seed(args.seed)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)

    # budild model
    encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    checkpoint_prefix = 'checkpoint-best-bleu/pytorch_model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir))
    logger.info("reload model from {}".format(output_dir))
    model.to(args.device)

    ## Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    codebert_mlm.to('cuda')

    file = "data/{}_statement.csv".format(args.statement)
    df = pd.read_csv(file, encoding="utf-8")
    # print(df.columns)
    code_list = df["code"].tolist()
    idx_list = df["idx"].tolist()
    identifier_list = df["identifier"].tolist()

    ## Load code
    eval_examples = read_examples(file)

    subs_path = "data/data_subs_test.jsonl"
    substitutes_list = {}
    with open(subs_path) as f:
        for line in f:
            js = json.loads(line.strip())
            substitutes_list[js["idx"]] = js["substitutes"]

    substitutes = []
    for idx in idx_list:
        substitutes.append(substitutes_list[idx])
    assert len(code_list) == len(substitutes)

    code_tokens = []
    for index, code in enumerate(code_list):
        code_tokens.append(get_identifiers(code, "java")[1])

    id2token, token2id = build_vocab(code_tokens, 5000)
    bleu_file = "mhm_" + args.statement
    if os.path.exists(bleu_file) is False:
        os.makedirs(bleu_file)
    attacker = MHM_Attacker(args, model, tokenizer, tokenizer_mlm, token2id, id2token, bleu_file)

    attack_succ_1 = 0
    attack_succ_3 = 0
    # recoder_1 = Recorder_summary("result/attack_mhm_" + args.statement + "_1.csv")
    # recoder_3 = Recorder_summary("result/attack_mhm_" + args.statement + "_3.csv")
    total_cnt = 0
    for index, example in enumerate(eval_examples):
        print("Index: ", index)
        original_bleu, pre_summary, ref_summary = eval_bleu(args, [example], model, tokenizer, bleu_file)
        if original_bleu[0] == 0:
            continue
        code = example.source
        identifiers, code_tokens = get_identifiers(code, 'java')
        if len(identifiers) == 0:
            continue
        substitute = substitutes[index]
        substitute_key = list(substitute.keys())

        statements = identifier_list[index].replace(" ", "").replace("'", "").strip('[').strip(']').split(',')
        statements = [] if statements == [''] else statements
        statements = [state for state in statements if state in substitute_key]
        statements = filter_state(code, statements)
        print("statements:", statements)
        if len(statements) == 0:
            continue

        total_cnt += 1
        if len(statements) == 1:
            result = attacker.mcmc_random(example, statements, substitute, tokenizer, code,
                                        _label=ref_summary, _n_candi=30,
                                        _max_iter=10, _prob_threshold=1)
            if result['is_success'] == 1:
                attack_succ_1 += 1
                attack_succ_3 += 1

        elif len(statements) <= 3:
            flag = 0
            statements_1 = random.sample(statements, 1)
            result_1 = attacker.mcmc_random(example, statements_1, substitute, tokenizer, code,
                                          _label=ref_summary, _n_candi=30,
                                          _max_iter=10, _prob_threshold=1)
            if result_1['is_success'] == 1:
                attack_succ_1 += 1
                attack_succ_3 += 1
                flag = 1
            if flag == 0:
                result = attacker.mcmc_random(example, statements, substitute, tokenizer, code,
                                          _label=ref_summary, _n_candi=30,
                                          _max_iter=10, _prob_threshold=1)
                if result['is_success'] == 1:
                    attack_succ_3 += 1

        elif len(statements) > 3:
            flag = 0
            statements_1 = random.sample(statements, 1)
            result_1 = attacker.mcmc_random(example, statements_1, substitute, tokenizer, code,
                                          _label=ref_summary, _n_candi=30,
                                          _max_iter=10, _prob_threshold=1)
            if result_1['is_success'] == 1:
                attack_succ_1 += 1
                attack_succ_3 += 1

                flag = 1
            if flag == 0:
                statements_3 = random.sample(statements, 3)
                result_3 = attacker.mcmc_random(example, statements_3, substitute, tokenizer, code,
                                              _label=ref_summary, _n_candi=30,
                                              _max_iter=10, _prob_threshold=1)
                if result_3['is_success'] == 1:
                    attack_succ_3 += 1

        print("attack_succ with 1: {}/{}={}".format(attack_succ_1, total_cnt, (attack_succ_1 / total_cnt)))
        print("attack_succ with 3: {}/{}={}".format(attack_succ_3, total_cnt, (attack_succ_3 / total_cnt)))

    print("Final attack_succ with 1: {}/{}={}".format(attack_succ_1, total_cnt, (attack_succ_1 / total_cnt)))
    print("Final attack_succ with 3: {}/{}={}".format(attack_succ_3, total_cnt, (attack_succ_3 / total_cnt)))

if __name__ == '__main__':
    main()
