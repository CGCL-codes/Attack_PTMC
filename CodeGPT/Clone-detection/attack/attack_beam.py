# coding=utf-8
'''For attacking CodeBERT models'''
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import sys
from datetime import datetime
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
from model import Model
from utils import set_seed, get_code_tokens
from utils import Recorder, CodeDataset
from run import TextDataset
from attacker import get_code_pairs
from beamAttacker import Beam_Attacker, get_statement_identifier, convert_examples_to_features
from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          GPT2Config, GPT2Model, GPT2Tokenizer)
from transformers import logging
logging.set_verbosity_error()
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning
warnings.filterwarnings("ignore")
MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'gpt2': (GPT2Config, GPT2Model, GPT2Tokenizer)
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
    parser.add_argument('--beam_size', type=int, default=3,
                        help="beam_size")

    args = parser.parse_args()

    device = torch.device("cuda")
    args.device = device

    # Set seed
    set_seed(args.seed)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    model = Model(model, config, tokenizer, args)

    load_model_path = '../saved_models/checkpoint-best-f1/model.bin'
    model.load_state_dict(torch.load(load_model_path))
    print("{} - reload model from {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), load_model_path))

    model.to(args.device)

    cpu_cont = 16
    ## Load Dataset
    pool = multiprocessing.Pool(cpu_cont)
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file, pool=pool)
    ## Load code pairs
    source_codes = get_code_pairs(args.eval_data_file, tokenizer, args)

    subs_path = "../../../Identifier-Study/clone/data/data_subs_test.jsonl"
    generated_substitutions1 = []
    generated_substitutions = []
    indexs = []
    with open(subs_path) as f:
        for line in f:
            js = json.loads(line.strip())
            indexs.append(js["idx"])
            generated_substitutions1.append(js["substitutes"])
    for source in source_codes:
        for i, idx in enumerate(indexs):
            if source[0] == idx:
                generated_substitutions.append(generated_substitutions1[i])
    assert len(source_codes) == len(eval_dataset) == len(generated_substitutions)

    success_attack = 0
    total_cnt = 0

    recoder = Recorder(args.csv_store_path)
    attacker = Beam_Attacker(args, model, tokenizer)
    start_time = time.time()
    query_times = 0
    for index, example in enumerate(eval_dataset):
        print("Index: ", index)
        example_start_time = time.time()
        code_pair = source_codes[index]
        logits, preds = model.get_results([example], args.eval_batch_size)
        orig_label = preds[0]
        orig_prob = logits[0]
        orig_prob = max(orig_prob)
        true_label = example[1].item()
        if not orig_label == true_label:
            continue
        substitutes = generated_substitutions[index]
        first_idx = int(code_pair[0])
        first_code = code_pair[2]

        code_2 = code_pair[3]
        code_2 = " ".join(code_2.split())
        words_2 = tokenizer.tokenize(code_2)

        orig_code_tokens = get_code_tokens(first_code)
        identifiers = list(substitutes.keys())
        statement_dict = get_statement_identifier(first_idx, identifiers)
        if len(identifiers) == 0:
            continue
        total_cnt += 1
        print(statement_dict)
        result = attacker.beam_attack(orig_prob, example, substitutes, code_pair, statement_dict, args.beam_size)
        example_end_time = (time.time() - example_start_time) / 60
        print("Example time cost: ", round(example_end_time, 2), "min")
        print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
        print("Query times in this attack: ", model.query - query_times)
        if result["succ"] == 1:
            success_attack += 1
            recoder.write(index, first_code, result["adv_code"], len(orig_code_tokens), len(identifiers),
                          result["replace_info"], model.query - query_times, example_end_time, result["type"])

            # temp_replace = " ".join(result["adv_code"].split())
            # temp_replace = tokenizer.tokenize(temp_replace)
            # new_feature = convert_examples_to_features(temp_replace, words_2, true_label, None, None, tokenizer,
            #                                            args, None)
            # new_example = CodeDataset([new_feature])
            # logits, preds = model.get_results([new_example[0]], args.eval_batch_size)
            # adv_label = preds[0]
            # print("attack predict label: ", adv_label)
            # if adv_label != true_label:
            #     print("true adv!")
            # else:
            #     print("false adv!")
        else:
            recoder.write(index, None, None,
                      len(orig_code_tokens), len(identifiers),
                      None, model.query - query_times, example_end_time, "0")
        query_times = model.query
        print("Success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))
    print("Final success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))


if __name__ == '__main__':
    main()