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

import json
import logging
import argparse
import warnings
import torch
import multiprocessing
import time
from run import InputFeatures, convert_examples_to_features
from model import Model
from run import TextDataset
from utils import CodeDataset
from utils import set_seed
from utils import Recorder_style
from attacker import Style_Attacker, get_code_pairs, get_transfered_code
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2Model, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2Model, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--base_model", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--csv_store_path1", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--use_ga", action='store_true',
                        help="Whether to GA-Attack.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")

    args = parser.parse_args()

    args.device = torch.device("cuda")
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

    checkpoint_prefix = 'checkpoint-best-f1/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)
    logger.info("reload model from {}".format(output_dir))

    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)

    # Load Dataset
    ## Load Dataset
    cpu_cont = 16
    pool = multiprocessing.Pool(cpu_cont)
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file, pool=pool)

    ## Load code pairs
    source_codes = get_code_pairs(args.eval_data_file, tokenizer, args)

    subs_path = "../../../dataset/preprocess/test_subs_clone.jsonl"
    labels = []
    code1_list = []
    with open(subs_path) as f:
        for line in f:
            js = json.loads(line.strip())
            code1_list.append(js['code1'])
            labels.append(js["label"])

    recoder1 = Recorder_style(args.csv_store_path1)
    # recoder3 = Recorder_style(args.csv_store_path2)
    # attacker = Style_Attacker(args, model, tokenizer, tokenizer_mlm)
    attacker = Style_Attacker(args, model, tokenizer)

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
                tmp_code = tokenizer.tokenize(tmp_code)

                tmp_feature = convert_examples_to_features(tmp_code, words_2, true_label, None, None,
                                                           tokenizer, args, None)
                new_dataset = CodeDataset([tmp_feature])
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
                # print("Attack 3 time cost: ", round((time.time() - t0) / 60, 2), "min")
                example_end_time = (time.time() - example_start_time) / 60
                if is_success == 1:
                    # print("Example time cost: ", round(example_end_time, 2), "min")
                    # print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
                    # print("Query times in this attack: ", query_times)
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

    print(
        "Success rate is : {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))


if __name__ == '__main__':
    main()
