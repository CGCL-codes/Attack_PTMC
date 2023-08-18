import numpy as np
from replace_and_camelSplit import replace_a_to_b,split_c_and_s
# from get_bleu import return_bleu
from get_encoder import return_encoder
import bleu
import torch
import os
import logging
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm, trange
# from PLBART.Clone_detection.code.representjs.data.util import Timer, normalize_program, EncodeAsIds

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)


from tree_sitter import Language, Parser

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
UNK='<unk>'


logger = logging.getLogger(__name__)




def get_example(item):
    code1, code2, label, tokenizer, args = item
    code1 = tokenizer.tokenize(code1)
    code2 = tokenizer.tokenize(code2)
    return convert_examples_to_features(code1, code2, label,  tokenizer, args)



class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label


def convert_examples_to_features(code1_tokens, code2_tokens, label,  tokenizer, args):
    # source
    code1_tokens = code1_tokens[:args.block_size - 2]
    code1_tokens = [tokenizer.bos_token] + code1_tokens + [tokenizer.eos_token]
    code2_tokens = code2_tokens[:args.block_size - 2]
    code2_tokens = [tokenizer.bos_token] + code2_tokens + [tokenizer.eos_token]

    code1_ids = tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = args.block_size - len(code1_ids)
    code1_ids += [tokenizer.pad_token_id] * padding_length

    code2_ids = tokenizer.convert_tokens_to_ids(code2_tokens)
    code2_length = len(code2_ids)
    padding_length = args.block_size - code2_length
    code2_ids += [tokenizer.pad_token_id] * padding_length

    source_tokens = code1_tokens + code2_tokens
    source_ids = code1_ids + code2_ids
    length = len(code1_ids) + code2_length
    return InputFeatures(source_tokens, source_ids, label)


class TextDataset(Dataset):
    def __init__(self, code1, code2, label, tokenizer, args, file_path='train', block_size=512, pool=None):
        postfix = file_path.split('/')[-1].split('.txt')[0]
        self.examples = []
        index_filename = file_path
        logger.info("Creating features from index file at %s ", index_filename)
        data = []
        # with open(index_filename) as f:
        #     for line in f:
        #         line = line.strip()
        #         url1, url2, label = line.split('\t')
        #         if url1 not in url_to_code or url2 not in url_to_code:
        #             continue
        if label == '0':
            label = 0
        else:
            label = 1
        data.append((code1, code2, label, tokenizer, args))
        if 'test' not in postfix:
            data = random.sample(data, int(len(data) * 0.1))

        self.examples = pool.map(get_example, tqdm(data, total=len(data)))
        if 'train' in postfix:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):

        return torch.tensor(self.examples[item].input_ids), \
               torch.tensor(self.examples[item].label)


def load_and_cache_examples(args, tokenizer, code1,code2,lable, evaluate=False, test=False, pool=None):
    dataset = TextDataset(code1,code2,lable, tokenizer, args, file_path=args.test_data_file if test else (
        args.eval_data_file if evaluate else args.train_data_file), block_size=args.block_size, pool=pool)
    return dataset


def computer_word_saliency_cos(model,code,code2,summary,variable_list,vocab,embeddings,max_token,
                               vocab_src,vocab_trg,max_token_src,max_token_trg):
    word_saliency_list=[]
    code_=split_c_and_s(code.split(' '))
    #_,encoder=return_bleu(code_,summary,model)

    encoder=return_encoder(code_,summary,model,vocab_src,vocab_trg,max_token_src,max_token_trg)

    for var in variable_list:
        var_index = vocab[var] if var in vocab else max_token
        embedding_var=embeddings[var_index]

        cos=cosine_similarity(embedding_var.reshape(1,64),encoder.reshape(1,64))[0][0]
        cos=(1.0+cos)*0.5
        word_saliency_list.append((var,cos))
    return word_saliency_list

def ids_to_strs(Y, sp):
    ids = []
    eos_id = sp.PieceToId("</s>")
    pad_id = sp.PieceToId("<pad>")
    for idx in Y:
        ids.append(int(idx))
        if int(idx) == eos_id or int(idx) == pad_id:
            break
    return sp.DecodeIds(ids)

def return_bleu(args, model, eval_dataset):
    query_cnt = 0
    orig_prob = []
    orig_label = []
    for index, example in enumerate(eval_dataset):
        orig_prob, orig_label = model.get_results([example], args.eval_batch_size)
        query_cnt += 1

    return query_cnt, orig_prob, orig_label

def computer_best_substitution(model,code,code2,lable,variable_list,nearest_k_dict,args, tokenizer, pool, device, epoch, count):
    query_cnt = 0
    tmp_cnt = 0
    best_substitution_list = []
    code_ = split_c_and_s(code.split(' '))
    tmp_code = code
    code2_ = split_c_and_s(code2.split(' '))
    # print(code_)
    # print("old_bleu")
    # print("code", code_)
    # print("code2", code2_)
    # print("lable", lable)
    lable = lable.replace("\n", "")
    # eval_examples = read_examples(count,code_,summary)
    eval_dataset = load_and_cache_examples(args, tokenizer, code_, code2_, lable, test=True,pool=pool)
    tmp_cnt, orig_prob, orig_label = return_bleu(args, model, eval_dataset)
    query_cnt += tmp_cnt
    new_variable_list=[]
    for var in variable_list:  # variable list  for one code
        new_variable_list.append(var)
        print("new_bleu")
        print("var",var)
        max_delta_bleu = 0
        nearest_k = nearest_k_dict[var]  # a list
        best_new_var = nearest_k[0]
        for new_var in nearest_k:
            print("new_var", new_var)
            new_code_list = replace_a_to_b(code, var, new_var)
            new_code = split_c_and_s(new_code_list)
            # print("new_code", new_code)
            eval_dataset = load_and_cache_examples(args, tokenizer, new_code, code2_, lable, test=True,pool=pool)
            tmp_cnt, orig_prob_, orig_label_ = return_bleu(args, model, eval_dataset)
            query_cnt += tmp_cnt
            # new_bleu,_=return_bleu(new_code,summary,model)
            # print("orig_prob_: ", orig_prob_)
            # print("orig_label_: ", orig_label_)
            if orig_label != orig_label_:
                best_new_var = new_var
                max_delta_bleu = 1
                break
            else:

                delta_bleu = orig_prob[0][orig_label_[0]] - orig_prob_[0][orig_label_[0]]
                if max_delta_bleu < delta_bleu:
                    max_delta_bleu = delta_bleu
                    best_new_var = new_var
        best_substitution_list.append((var, best_new_var, max_delta_bleu))

        tmp_code_list = replace_a_to_b(tmp_code, var, best_new_var)
        tmp_code = split_c_and_s(tmp_code_list)

        eval_dataset = load_and_cache_examples(args, tokenizer, tmp_code, code2_, lable, test=True, pool=pool)
        tmp_cnt, orig_prob_, orig_label_ = return_bleu(args, model, eval_dataset)
        query_cnt += tmp_cnt
        if orig_label_ != orig_label:
            break


    return best_substitution_list, query_cnt, new_variable_list