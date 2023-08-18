import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from word_saliency_CodeBert import computer_best_substitution,computer_word_saliency_cos
from main.model import Code2NaturalLanguage   #main   return encoder vector
from gensim.models.word2vec import Word2Vec
import torch
import random
import logging
import argparse
import bleu
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
torch.cuda.current_device()
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import time
import numpy as np
from encoder.rnnModel import Seq2Seq
import multiprocessing
from CodeBert.Clone_detection.code.model import Model
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def softmax(x):
    exp_x=np.exp(x)
    softmax_x=exp_x/np.sum(exp_x)
    return softmax_x

def get_topk_index(k,arr):
    top_k=k
    array=arr
    top_k_index=array.argsort()[::-1][0:top_k]
    return top_k_index

def rank_variable(code,code2,summary,variable_list,nearest_k_dict,vocab,embeddings,max_token,
                  model_encoder,vocab_src,vocab_trg,max_token_src,max_token_trg):





    #print(word_saliency_list)
    # print("model in func (rank_variable)",model)
    best_substitution_list, tmp_cnt, new_variable_list=computer_best_substitution(model,code,code2,summary,variable_list,nearest_k_dict,args, tokenizer, pool, device, epoch,count)
    word_saliency_list = computer_word_saliency_cos(model_encoder, code,code2, summary, new_variable_list, vocab,embeddings,max_token,
                                                    vocab_src, vocab_trg, max_token_src, max_token_trg
                                                    )
    #print(best_substitution_list)
    unk_delta_bleu=[]
    best_delta_bleu=[]
    best_sub_list=[]
    for item in word_saliency_list:
        unk_delta_bleu.append(item[1])
    for item in best_substitution_list:
        best_delta_bleu.append(item[2])
        best_sub_list.append(item[1])

    np_unk_delta_bleu=np.array(unk_delta_bleu)
    np_best_delta_bleu=np.array(best_delta_bleu)
    np_unk_delta_bleu_soft=softmax(np_unk_delta_bleu)
    sorce=np_unk_delta_bleu_soft*np_best_delta_bleu

    for i in range(len(sorce)):
        if sorce[i]==0 and np_unk_delta_bleu_soft[i]!=0:
            sorce[i]=np_unk_delta_bleu_soft[i]*0.5
        if sorce[i]==0 and np_best_delta_bleu[i]!=0:
            sorce[i]=np_best_delta_bleu[i]

    print(sorce)
    descend_index=get_topk_index(len(sorce),sorce)
    print(descend_index)
    descend_variable={}
    for item in descend_index:
        var=variable_list[item]
        sub=best_sub_list[item]
        descend_variable[var]=sub
    print(descend_variable)
    return descend_variable, tmp_cnt

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def prepare():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_type", default="", type=str, )
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

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
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    cpu_cont = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_cont)
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size // args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
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
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    checkpoint_prefix = 'checkpoint-best-f1/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)

    epoch = -1
    # test(args, model, tokenizer, pool=pool, best_threshold=0.5)

    return args, model, tokenizer, device, epoch, pool, 0.5


if __name__=='__main__':
    totaltime = 0
    totaltime2 = 0
    totalquery = 0
    query_list= []
    time_list= []
    start_num = 11
    end_num = 12
    for mycnt in range(start_num, end_num):
        query_cnt = 0
        start = time.process_time()
        start2 = time.time()
        print('start at time:')
        print(time.strftime(' %Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        data_root = '../Clone-detection'
        f_code = open(data_root+'/data1/java' + str(mycnt) + '/test/code.original', 'r')
        f_code2 = open(data_root+'/data1/java' + str(mycnt) + '/test/code2.original', 'r')
        f_summary = open(data_root+'/data1/java' + str(mycnt) + '/test/javadoc.original', 'r')
        nearest_k_data = pd.read_pickle(
            # data_root+'/data/python/nearest_k_for_everyVar_test.pkl')
            data_root+'/var_name/java' + str(mycnt) + '/code_nearest_top30.pkl')
        var_everyCode_data = pd.read_pickle(
            data_root+'/var_for_everyCode_test' + str(mycnt) + '.pkl')

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
        model_encoder = Seq2Seq(embeddings_src,embeddings_trg)
        # model_encoder.load_state_dict(torch.load('/encoder/model/python_model.pkl'))
        model_encoder.load_state_dict(torch.load(data_root+'/encoder/model1.pkl'))
        #---------------------------------------------

        args, model, tokenizer, device, epoch ,pool , best_threshold= prepare()
        best_descend_var_list=[]
        count=0
        save=0
        for code, code2, summ in zip(f_code, f_code2, f_summary):
            # if count > 2000:
            #     break
            t0 = time.time()
            variable_list=var_everyCode_list[count]
            nearest_k_dict=nearest_k_list[count]
            print("rank_variable")
            descend_variable_dict, f_tmp_cnt=rank_variable(code,code2,summ,variable_list,nearest_k_dict,vocab,embeddings,max_token,
                                                model_encoder,vocab_src,vocab_trg,max_token_src,max_token_trg)
            t1 = time.time()
            query_cnt += f_tmp_cnt
            query_list.append(f_tmp_cnt)
            time_list.append(t1-t0)
            best_descend_var_list.append(descend_variable_dict)
            count=count+1
            print('ok'+str(count))
            '''
    
            if count%5000==0:
                index = [i for i in range(len(best_descend_var_list))]
                data_descend_best_var = pd.DataFrame({'id': index, 'var_sub': best_descend_var_list})
                data_descend_best_var.to_pickle(
                    '/home/zxq/code/attacker_transformer/attacker/var_name/python/train/trans/train_trans_data_descend_best_var_8.pkl')
                save=save+1
                print('saved')
            '''

        index=[i for i in range(len(best_descend_var_list))]
        data_descend_best_var=pd.DataFrame({'id':index,'var_sub':best_descend_var_list})
        data_descend_best_var.to_pickle(
        #     # data_root + '/data/python/test_rnn_data_descend_best_var.pkl')
            data_root + '/CodeBert/data/java' + str(mycnt) + '/test_rnn_data_descend_best_var.pkl')
        print('end...')
        print(time.strftime(' %Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        end = time.process_time()
        end2 = time.time()
        totaltime += end - start
        totaltime2 += end2 - start2
        totalquery += query_cnt
        print("time cost: ", end - start)
        print("time cost2: ", end2 - start2)
        print("query_cnt: ", query_cnt)




