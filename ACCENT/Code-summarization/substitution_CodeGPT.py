import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from word_saliency_CodeGPT import computer_best_substitution, computer_word_saliency_cos
from main.model import Code2NaturalLanguage  # main   return encoder vector
from gensim.models.word2vec import Word2Vec
import torch
import random
import logging
import argparse
import bleu

import torch.nn as nn

torch.cuda.current_device()
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import time
import numpy as np
from encoder.rnnModel import Seq2Seq
# from model import Seq2Seq_
from CodeGPT.code.model import Seq2Seq_
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
                 }
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def get_topk_index(k, arr):
    top_k = k
    array = arr
    top_k_index = array.argsort()[::-1][0:top_k]
    return top_k_index


def rank_variable(code, summary, variable_list, nearest_k_dict, vocab, embeddings, max_token,
                  model_encoder, vocab_src, vocab_trg, max_token_src, max_token_trg):

    # print(word_saliency_list)
    # print("model in func (rank_variable)",model)
    best_substitution_list, tmp_cnt,new_variable_list = computer_best_substitution(model, code, summary, variable_list, nearest_k_dict,
                                                                 args, tokenizer, device, epoch, count)
    word_saliency_list = computer_word_saliency_cos(model_encoder, code, summary, new_variable_list, vocab, embeddings,
                                                    max_token,
                                                    vocab_src, vocab_trg, max_token_src, max_token_trg
                                                    )
    # print(best_substitution_list)
    unk_delta_bleu = []
    best_delta_bleu = []
    best_sub_list = []
    for item in word_saliency_list:
        unk_delta_bleu.append(item[1])
    for item in best_substitution_list:
        best_delta_bleu.append(item[2])
        best_sub_list.append(item[1])

    np_unk_delta_bleu = np.array(unk_delta_bleu)
    np_best_delta_bleu = np.array(best_delta_bleu)
    np_unk_delta_bleu_soft = softmax(np_unk_delta_bleu)
    sorce = np_unk_delta_bleu_soft * np_best_delta_bleu

    for i in range(len(sorce)):
        if sorce[i] == 0 and np_unk_delta_bleu_soft[i] != 0:
            sorce[i] = np_unk_delta_bleu_soft[i] * 0.5
        if sorce[i] == 0 and np_best_delta_bleu[i] != 0:
            sorce[i] = np_best_delta_bleu[i]

    print(sorce)
    descend_index = get_topk_index(len(sorce), sorce)
    print(descend_index)
    descend_variable = {}
    for item in descend_index:
        var = variable_list[item]
        sub = best_sub_list[item]
        descend_variable[var] = sub
    print(descend_variable)
    return descend_variable, tmp_cnt


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def update_config(model, tokenizer):
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id


def prepare():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--stop_no_improve_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--lang", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    # print arguments
    args = parser.parse_args()

    # args.model_name_or_path = "microsoft/codebert-base"
    args.model_name_or_path = "microsoft/CodeGPT-small-java-adaptedGPT2"
    args.load_model_path = "./CodeGPT/saved_models/checkpoint-best-bleu/pytorch_model.bin"
    args.model_type = "gpt2"
    args.output_dir = "./CodeGPT/saved_models/"
    args.lang = "java"
    # args.test_filename = "../../../../dataset/Code-summarization/test_1.jsonl"
    args.seed = 123456
    args.max_source_length = 256
    args.max_target_length = 128
    args.beam_size = 10
    args.train_batch_size = 32
    args.eval_batch_size = 32
    logger.info(args)

    # Setup CUDA, GPU & distributed training

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
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
    model = Seq2Seq_(decoder=decoder, config=decoder.config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.bos_token_id, eos_id=tokenizer.eos_token_id)
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    model = torch.nn.DataParallel(model)

    epoch = -1

    return args, model, tokenizer, device, epoch
    # test(args, model, tokenizer, device, epoch)


if __name__ == '__main__':
    totaltime = 0
    totaltime2 = 0
    totalquery = 0
    query_list = []
    time_list = []
    start_num = 11
    end_num = 12
    for mycnt in range(start_num, end_num):
        query_cnt = 0
        start = time.process_time()
        start2 = time.time()
        print('start at time:')
        print(time.strftime(' %Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        data_root = '../Code-summarization'
        f_code = open(data_root + '/data1/java' + str(mycnt) + '/test/code.original', 'r')
        f_summary = open(data_root + '/data1/java' + str(mycnt) + '/test/javadoc.original', 'r')
        nearest_k_data = pd.read_pickle(
            # data_root+'/data/python/nearest_k_for_everyVar_test.pkl')
            data_root + '/var_name/java' + str(mycnt) + '/code_nearest_top30.pkl')
        var_everyCode_data = pd.read_pickle(
            data_root + '/var_for_everyCode_test' + str(mycnt) + '.pkl')

        nearest_k_list = nearest_k_data['nearest_k'].tolist()
        var_everyCode_list = var_everyCode_data['variable'].tolist()
        print('read embedding....')
        root = ' '
        # word2vec = Word2Vec.load('/embedding/python/node_w2v_64').wv
        word2vec = Word2Vec.load(data_root + '/data1/java1/node_w2v_64').wv
        vocab = word2vec.key_to_index
        max_token = word2vec.vectors.shape[0]
        embedding_dim = word2vec.vectors.shape[1]
        embeddings = np.zeros((max_token + 1, embedding_dim))
        embeddings[:max_token] = word2vec.vectors
        print('end read embedding..')

        print('read embedding of src and trg....')
        word2vec_src = Word2Vec.load(
            # '/encoder/vocab/python/python_node_w2v_code_64').wv
            data_root + '/encoder/vocab/train1/node_w2v_code_64').wv
        vocab_src = word2vec_src.key_to_index
        max_token_src = word2vec_src.vectors.shape[0]
        embedding_dim = word2vec_src.vectors.shape[1]
        embeddings_src = np.zeros((max_token_src + 1, embedding_dim))
        embeddings_src[:max_token_src] = word2vec_src.vectors

        word2vec_trg = Word2Vec.load(
            # '/encoder/vocab/python/python_node_w2v_summ_64').wv
            data_root + '/encoder/vocab/train1/node_w2v_summ_64').wv
        vocab_trg = word2vec_trg.key_to_index
        max_token_trg = word2vec_trg.vectors.shape[0]
        embedding_dim = word2vec_trg.vectors.shape[1]
        embeddings_trg = np.zeros((max_token_trg + 1, embedding_dim))
        embeddings_trg[:max_token_trg] = word2vec_trg.vectors

        print('load encoder model:')
        model_encoder = Seq2Seq(embeddings_src, embeddings_trg)
        # model_encoder.load_state_dict(torch.load('/encoder/model/python_model.pkl'))
        model_encoder.load_state_dict(torch.load(data_root + '/encoder/model1.pkl'))
        # ---------------------------------------------

        args, model, tokenizer, device, epoch = prepare()
        best_descend_var_list = []
        count = 0
        save = 0
        for code, summ in zip(f_code, f_summary):
            t0 = time.time()
            variable_list = var_everyCode_list[count]
            nearest_k_dict = nearest_k_list[count]
            print("rank_variable")
            descend_variable_dict, f_tmp_cnt = rank_variable(code, summ, variable_list, nearest_k_dict, vocab,
                                                             embeddings, max_token,
                                                             model_encoder, vocab_src, vocab_trg, max_token_src,
                                                             max_token_trg)
            t1 = time.time()
            query_cnt += f_tmp_cnt
            query_list.append(f_tmp_cnt)
            time_list.append(t1 - t0)
            best_descend_var_list.append(descend_variable_dict)
            count = count + 1
            print('ok' + str(count))
            '''

            if count%5000==0:
                index = [i for i in range(len(best_descend_var_list))]
                data_descend_best_var = pd.DataFrame({'id': index, 'var_sub': best_descend_var_list})
                data_descend_best_var.to_pickle(
                    '/home/zxq/code/attacker_transformer/attacker/var_name/python/train/trans/train_trans_data_descend_best_var_8.pkl')
                save=save+1
                print('saved')
            '''

        index = [i for i in range(len(best_descend_var_list))]
        data_descend_best_var = pd.DataFrame({'id': index, 'var_sub': best_descend_var_list})
        data_descend_best_var.to_pickle(
            # data_root + '/data/python/test_rnn_data_descend_best_var.pkl')
            data_root + '/CodeGPT/data/java' + str(mycnt) + '/test_rnn_data_descend_best_var_12_16.pkl')
        print('end...')
        print(time.strftime(' %Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        end = time.process_time()
        end2 = time.time()
        totaltime += end - start
        totaltime2 += end2 - start2
        totalquery += query_cnt
        print("time cost2: ", end2 - start2)
        print("query_cnt: ", query_cnt)

