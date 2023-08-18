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
import logging
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
from model import Model
from utils import set_seed
import sentencepiece as spm
from utils import Recorder, get_code_tokens
from run import TextDataset
from attacker import get_code_pairs
from attacker import ALERT_Attacker
from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
from fairseq.models.bart import BARTModel

warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning
logger = logging.getLogger(__name__)
class Dictionary():
    def __init__(self, vocab_size, pad_id) -> None:
        self.vocab_size = vocab_size
        self.pad_id = pad_id

    def __len__(self):
        return self.vocab_size

    def pad(self):
        return self.pad_id
class Task():
    def __init__(self, vocab_size, pad_id) -> None:
        self.source_dictionary = Dictionary(vocab_size, pad_id)
        self.target_dictionary = self.source_dictionary
def build_mbart(args):
    task = Task(args.vocab_size, args.pad_id)
    model = BARTModel.build_model(args, task)
    return model

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
    parser.add_argument('--start_num', type=int, default=0,
                        help="start_num")
    parser.add_argument('--end_num', type=int, default=3999,
                        help="end_num")

    args = parser.parse_args()

    device = torch.device("cuda")
    args.device = device

    # Set seed
    set_seed(args.seed)

    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer_name)
    config = argparse.Namespace(activation_fn='gelu', adam_betas='(0.9, 0.98)', adam_eps=1e-08, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, add_prev_output_tokens=True, all_gather_list_size=16384, arch='mbart_base', attention_dropout=0.1, batch_size=4, batch_size_valid=4, best_checkpoint_metric='accuracy', bf16=False, bpe=None, broadcast_buffers=False, bucket_cap_mb=25, checkpoint_shard_count=1, checkpoint_suffix='', classification_head_name='sentence_classification_head', clip_norm=1.0, cpu=False, criterion='sentence_prediction', cross_self_attention=False, curriculum=0, data='/home/zzr/CodeStudy/Defect-detection/plbart/processed/data-bin', data_buffer_size=10, dataset_impl=None, ddp_backend='no_c10d', decoder_attention_heads=12, decoder_embed_dim=768, decoder_embed_path=None, decoder_ffn_embed_dim=3072, decoder_input_dim=768, decoder_layerdrop=0, decoder_layers=6, decoder_layers_to_keep=None, decoder_learned_pos=True, decoder_normalize_before=False, decoder_output_dim=768, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=False, distributed_num_procs=0, distributed_port=-1, distributed_rank=0, distributed_world_size=1, distributed_wrapper='DDP', dropout=0.1, empty_cache_freq=0, encoder_attention_heads=12, encoder_embed_dim=768, encoder_embed_path=None, encoder_ffn_embed_dim=3072, encoder_layerdrop=0, encoder_layers=6, encoder_layers_to_keep=None, encoder_learned_pos=True, encoder_normalize_before=False, end_learning_rate=0.0, fast_stat_sync=False, find_unused_parameters=True, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=None, force_anneal=None, fp16=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test', init_token=0, keep_best_checkpoints=-1, keep_interval_updates=-1, keep_last_epochs=-1, langs='java,python,en_XX', layernorm_embedding=True, local_rank=0, localsgd_frequency=3, log_format='json', log_interval=10, lr=[5e-05], lr_scheduler='polynomial_decay', max_epoch=5, max_positions=512, max_source_positions=1024, max_target_positions=1024, max_tokens=2048, max_tokens_valid=2048, max_update=15000, maximize_best_checkpoint_metric=True, memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, min_lr=-1.0, model_parallel_size=1, no_cross_attention=False, no_epoch_checkpoints=True, no_last_checkpoints=False, no_progress_bar=False, no_save=False, no_save_optimizer_state=False, no_scale_embedding=False, no_seed_provided=False, no_shuffle=False, no_token_positional_embeddings=False, nprocs_per_node=1, num_classes=2, num_shards=1, num_workers=1, optimizer='adam', optimizer_overrides='{}', patience=-1, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, pooler_activation_fn='tanh', pooler_dropout=0.0, power=1.0, profile=False, quant_noise_pq=0, quant_noise_pq_block_size=8, quant_noise_scalar=0, quantization_config_path=None, regression_target=False, relu_dropout=0.0, required_batch_size_multiple=1, required_seq_len_multiple=1, reset_dataloader=True, reset_lr_scheduler=False, reset_meters=True, reset_optimizer=True, restore_file='/data2/cg/CodeStudy/PLBART/pretrain/checkpoint_11_100000.pt', save_dir='/home/zzr/CodeStudy/Defect-detection/plbart/devign', save_interval=1, save_interval_updates=0, scoring='bleu', seed=1234, sentence_avg=False, separator_token=None, shard_id=0, share_all_embeddings=True, share_decoder_input_output_embed=True, shorten_data_split_list='', shorten_method='truncate', skip_invalid_size_inputs_valid_test=False, slowmo_algorithm='LocalSGD', slowmo_momentum=None, stop_time_hours=0, task='plbart_sentence_prediction', tensorboard_logdir=None, threshold_loss_scale=None, tokenizer=None, total_num_update=1000000, tpu=False, train_subset='train', update_freq=[4], use_bmuf=False, use_old_adam=False, user_dir='/home/zzr/CodeStudy/PLBART/source', valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, warmup_updates=500, weight_decay=0.0, zero_sharding='none')
    config.num_labels = 1
    config.vocab_size = tokenizer.GetPieceSize() + 5
    config.pad_id = 1
    assert config.vocab_size == 50005

    max_len_single_sentence = 510
    if args.block_size <= 0:
        args.block_size = max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, max_len_single_sentence)

    model = build_mbart(config)
    if args.model_name_or_path:
        sd = torch.load(args.model_name_or_path, map_location="cpu")
        model.load_state_dict(sd["model"])
        logger.info("Reload model from {}.".format(args.model_name_or_path))

    model = Model(model, config, tokenizer, args)

    load_model_path = '../saved_models/checkpoint-best-f1/model.bin'
    model.load_state_dict(torch.load(load_model_path))
    print("{} - reload model from {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), load_model_path))

    model.to(args.device)

    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)

    ## Load Dataset
    cpu_cont = 16
    pool = multiprocessing.Pool(cpu_cont)
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file, pool=pool)
    ## Load code pairs
    source_codes = get_code_pairs(args.eval_data_file, tokenizer, args)

    subs_path = "../../../dataset/preprocess/test_subs_clone.jsonl"
    generated_substitutions = []
    with open(subs_path) as f:
        for line in f:
            js = json.loads(line.strip())
            generated_substitutions.append(js["substitutes"])
    assert len(source_codes) == len(eval_dataset) == len(generated_substitutions)

    success_attack = 0
    total_cnt = 0

    recoder = Recorder(args.csv_store_path)
    attacker = ALERT_Attacker(args, model, tokenizer, tokenizer_mlm, use_bpe=1, threshold_pred_score=0)
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
        true_label = code_pair[4]
        if not orig_label == true_label:
            continue
        substitutes = generated_substitutions[index]
        first_code = code_pair[2]
        orig_code_tokens = get_code_tokens(first_code)
        identifiers = list(substitutes.keys())
        if len(identifiers) == 0:
            continue
        total_cnt += 1

        print(identifiers)
        code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attacker.greedy_attack(
            example, substitutes, code_pair)
        attack_type = "Greedy"
        if is_success == -1 and args.use_ga:
            code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attacker.ga_attack(
                example, substitutes, code, initial_replace=replaced_words)
            attack_type = "GA"

        example_end_time = (time.time() - example_start_time) / 60
        print("Example time cost: ", round(example_end_time, 2), "min")
        print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
        print("Query times in this attack: ", model.query - query_times)
        replace_info = ''
        if replaced_words is not None:
            for key in replaced_words.keys():
                replace_info += key + ':' + replaced_words[key] + ','

        if is_success == 1:
            success_attack += 1
            recoder.write(index, first_code, adv_code, len(orig_code_tokens),
                          len(identifiers),
                          replace_info, model.query - query_times, example_end_time, attack_type)
        else:
            recoder.write(index, None, None, len(orig_code_tokens), len(identifiers),
                          None, model.query - query_times, example_end_time, "0")
        query_times = model.query
        print("Success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))
    print("Final success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))


if __name__ == '__main__':
    main()