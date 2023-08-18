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
from run_parser import get_identifiers, get_example
from utils import set_seed, build_vocab
from model import Seq2Seq
from utils import Recorder, get_code_tokens
from run import TextDataset, convert_examples_to_features
from attacker import WIR_Attacker, read_examples
from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset


warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)
}

logger = logging.getLogger(__name__)

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

    args = parser.parse_args()

    device = torch.device("cuda")
    args.device = device

    # Set seed
    set_seed(args.seed)

    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    # budild model
    load_model_path = "../saved_models/checkpoint-best-bleu/pytorch_model.bin"
    encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    if load_model_path is not None:
        logger.info("reload model from {}".format(load_model_path))
        model.load_state_dict(torch.load(load_model_path))

    model.to(device)

    tokenizer_mlm = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

    ## Load code
    eval_examples = read_examples(args.dev_filename)
    code_tokens = []
    for index, example in enumerate(eval_examples):
        source_code = example.source
        code_tokens.append(get_identifiers(source_code, "java")[1])

    id2token, token2id = build_vocab(code_tokens, 5000)
    bleu_file = "wir"
    if os.path.exists(bleu_file) is False:
        os.makedirs(bleu_file)
    recoder = Recorder(args.csv_store_path)
    attacker = WIR_Attacker(args, model, tokenizer, token2id, id2token, bleu_file)

    success_attack = 0
    total_cnt = 0
    start_time = time.time()
    query_times = 0
    for index, example in enumerate(eval_examples):
        print("Index: ", index)
        original_bleu, pre_summary, ref_summary = attacker.eval_bleu(example)
        if original_bleu == 0:
            continue
        code = example.source
        orig_code_tokens = get_code_tokens(code)
        identifiers, _ = get_identifiers(code, 'java')
        if len(identifiers) == 0:
            continue
        identifiers = [iden[0] for iden in identifiers]
        total_cnt += 1

        example_start_time = time.time()
        is_success, adv_code, adv_summary, nb_changed_var, nb_changed_pos, replaced_words = attacker.wir_random_attack(
            original_bleu, example)

        example_end_time = (time.time() - example_start_time) / 60
        replace_info = ''
        if replaced_words is not None:
            for key in replaced_words.keys():
                replace_info += key + ':' + replaced_words[key] + ','
        print("Example time cost: ", round(example_end_time, 2), "min")
        print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
        print("Query times in this attack: ", attacker.query - query_times)
        if is_success == 1:
            success_attack += 1
            recoder.write(index, code, adv_code, len(orig_code_tokens), len(identifiers), replace_info,
                          attacker.query - query_times, round(example_end_time, 2), "WIR")
        else:
            recoder.write(index, None, None, len(orig_code_tokens), len(identifiers),
                          None, attacker.query - query_times, example_end_time, "0")
        query_times = attacker.query
        print("Success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))
    print("Final success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))

if __name__ == '__main__':
    main()