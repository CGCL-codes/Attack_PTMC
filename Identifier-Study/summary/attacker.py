import sys
import os
import subprocess
sys.path.append('../../CodeBERT/Code-summarization/code')
sys.path.append('../../../')
sys.path.append('../code')
sys.path.append('../../../python_parser')
import csv
import copy
import json
import argparse
import warnings
import torch
import numpy as np
import random
import time
import bleu
from run import TextDataset, InputFeatures, convert_examples_to_features
from utils import select_parents, crossover, map_chromesome, mutate, is_valid_variable_name, _tokenize, \
    get_identifier_posistions_from_code, get_masked_code_by_position, get_substitues, is_valid_substitue, set_seed

from utils import CodeDataset
from utils import getUID, isUID, getTensor, build_vocab
from run_parser import get_identifiers, get_example
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

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


def get_new_example(idx, code, nl):
    examples = []
    examples.append(
        Example(idx=idx, source=code, target=nl, )
    )
    return examples

def eval_bleu(args, examples, model, tokenizer, bleu_file):
    ref_summary = []
    for examp in examples:
        ref_summary.append(examp.target)
    eval_features = convert_examples_to_features(examples, tokenizer, args, stage='test')
    eval_data = TextDataset(eval_features, args)

    # Calculate bleu
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pre=[]
    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids, source_mask, target_mask = batch
        with torch.no_grad():
            preds = model(source_ids=source_ids,source_mask=source_mask)
            for pred in preds:
                t=pred[0].cpu().numpy()
                t=list(t)
                if 0 in t:
                    t=t[:t.index(0)]
                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                pre.append(text)

    pre_summary = pre
    model.train()
    bleus = []
    for p, example in zip(pre, examples):
        predictions=[]
        if os.path.exists(bleu_file + "/dev.output"):
            os.remove(bleu_file + "/dev.output")
        if os.path.exists(bleu_file + "/dev.gold"):
            os.remove(bleu_file + "/dev.gold")
        with open((bleu_file + "/dev.output"),'w') as f, open((bleu_file + "/dev.gold"),'w') as f1:
            predictions.append(str(example.idx)+'\t'+p)
            f.write(str(example.idx)+'\t'+p+'\n')
            f1.write(str(example.idx)+'\t'+example.target+'\n')
        try:
            (goldMap, predictionMap) = bleu.computeMaps(predictions, bleu_file + "/dev.gold")
            dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
        except:
            dev_bleu = -1
        bleus.append(dev_bleu)

    return bleus, pre_summary, ref_summary

def get_importance_score(args, bleu_file, idx, nl, code, current_prob, words_list: list, variable_names: list, model, tokenizer):
    '''Compute the importance score of each variable'''
    # label: example[1] tensor(1)
    # 1. 过滤掉所有的keywords.
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    # 需要注意大小写.
    if len(positions) == 0:
        ## 没有提取出可以mutate的position
        return None, None, None

    new_bleus = []

    # 2. 得到Masked_tokens
    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)
    # replace_token_positions 表示着，哪一个位置的token被替换了.

    for index, tokens in enumerate([words_list] + masked_token_list):
        new_code = ' '.join(tokens)
        new_example = get_new_example(idx, new_code, nl)
        new_bleu, _, _ = eval_bleu(args, new_example, model, tokenizer, bleu_file)
        new_bleus.append(new_bleu)

    importance_score = []
    for new_bleu in new_bleus:
        importance_score.append(current_prob - new_bleu[0])

    return importance_score, replace_token_positions, positions

class MHM_Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, tokenizer_mlm, _token2idx, _idx2token, bleu_file) -> None:
        self.classifier = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        self.args = args
        self.tokenizer_mlm = tokenizer_mlm
        self.bleu_file = bleu_file

    def mcmc_random(self, example, statements, substitutions, tokenizer, code, _label=None, _n_candi=30,
                    _max_iter=100, _prob_threshold=0.95):
        idx = example.idx
        nl = example.target
        identifiers, code_tokens = get_identifiers(code, 'java')
        processed_code = " ".join(code_tokens)
        prog_length = len(code_tokens)
        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)
        raw_tokens = copy.deepcopy(words)
        variable_names = statements

        uid = get_identifier_posistions_from_code(words, variable_names)

        if len(uid) <= 0:  # 是有可能存在找不到变量名的情况的.
            return {'succ': None, 'tokens': None, 'raw_tokens': None, "is_success": -1}

        variable_substitue_dict = {}
        for tgt_word in uid.keys():
            variable_substitue_dict[tgt_word] = substitutions[tgt_word]
        old_uids = {}
        old_uid = ""
        final_query_times = 0
        for iteration in range(1, 1 + _max_iter):
            # 这个函数需要tokens
            res = self.__replaceUID_random(idx, nl, _tokens=code, _label=_label, _uid=uid,
                                           substitute_dict=variable_substitue_dict,
                                           _n_candi=_n_candi,
                                           _prob_threshold=_prob_threshold)

            final_query_times += res["query_times"]
            if res and res['status'].lower() == 's':
                # self.__printRes(_iter=iteration, _res=res, _prefix="  >> ")
                if iteration == 1:
                    old_uids[res["old_uid"]] = []
                    old_uids[res["old_uid"]].append(res["new_uid"])
                    old_uid = res["old_uid"]

                flag = 0
                for k in old_uids.keys():
                    if res["old_uid"] == old_uids[k][-1]:
                        flag = 1
                        old_uids[k].append(res["new_uid"])
                        old_uid = k
                        break
                if flag == 0:
                    old_uids[res["old_uid"]] = []
                    old_uids[res["old_uid"]].append(res["new_uid"])
                    old_uid = res["old_uid"]

                code = res['tokens']
                uid[res['new_uid']] = uid.pop(res['old_uid'])  # 替换key，但保留value.
                variable_substitue_dict[res['new_uid']] = variable_substitue_dict.pop(res['old_uid'])

                for i in range(len(raw_tokens)):
                    if raw_tokens[i] == res['old_uid']:
                        raw_tokens[i] = res['new_uid']
                if res['status'].lower() == 's':
                    replace_info = {}
                    nb_changed_pos = 0
                    for uid_ in old_uids.keys():
                        replace_info[uid_] = old_uids[uid_][-1]
                        nb_changed_pos += len(uid[old_uids[uid_][-1]])
                    return {'succ': True, 'tokens': code,
                            "adv_summary": res["adv_summary"],
                            "is_success": 1, "old_uid": old_uid, "query_times": final_query_times,
                            "nb_changed_var": len(old_uids), "nb_changed_pos": nb_changed_pos,
                            "replace_info": replace_info}

        return {'succ': False, "is_success": -1}

    def __replaceUID_random(self, idx, nl, _tokens, _label=None, _uid={}, substitute_dict={},
                            _n_candi=30, _prob_threshold=0.95, _candi_mode="random"):

        assert _candi_mode.lower() in ["random", "nearby"]
        query_times = 0
        selected_uid = random.sample(substitute_dict.keys(), 1)[0]  # 选择需要被替换的变量名
        if _candi_mode == "random":
            # First, generate candidate set.
            # The transition probabilities of all candidate are the same.
            candi_token = [selected_uid]
            candi_tokens = [copy.deepcopy(_tokens)]
            candi_labels = [_label]
            for c in random.sample(self.idx2token, _n_candi):  # 选出_n_candi数量的候选.
                if c in _uid.keys():
                    continue
                if isUID(c):  # 判断是否是变量名.
                    candi_token.append(c)
                    candi_tokens.append(copy.deepcopy(_tokens))
                    candi_labels.append(_label)
                    candi_tokens[-1] = get_example(candi_tokens[-1], selected_uid, c)
                    # for i in _uid[selected_uid]: # 依次进行替换.
                    #     if i >= len(candi_tokens[-1]):
                    #         break
                    #     candi_tokens[-1][i] = c # 替换为新的candidate.
            pred_list = []
            query_times += len(candi_tokens)
            new_examples = []
            for i, tmp_tokens in enumerate(candi_tokens):
                tmp_code = tmp_tokens
                new_example = get_new_example(idx, tmp_code, nl)
                new_examples.append(new_example[0])

            bleus, pre_summary, ref_summary = eval_bleu(self.args, new_examples, self.classifier, self.tokenizer_tgt, self.bleu_file)
            for i, bleu in enumerate(bleus):
                if bleu == 0.0:  # 如果有样本攻击成功
                    return {"status": "s", "alpha": 1, "tokens": candi_tokens[i],
                            "old_uid": selected_uid, "new_uid": candi_token[i], "query_times": query_times,
                            "adv_summary": pre_summary[i], "nb_changed_pos": _tokens.count(selected_uid)}


            return {"status": "f", "query_times": query_times}

    def __printRes(self, _iter=None, _res=None, _prefix="  => "):
        if _res['status'].lower() == 's':  # Accepted & successful
            print("%s iter %d, SUCC! %s => %s" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid']), flush=True)

class WIR_Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, tokenizer_mlm, _token2idx, _idx2token, bleu_file) -> None:
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        self.args = args
        self.tokenizer_mlm = tokenizer_mlm
        self.bleu_file = bleu_file

    def wir_random_attack(self, current_prob, statement, example):
        # Start the attack
        code = example.source
        idx = example.idx
        nl = example.target
        original_bleu = current_prob
        identifiers, code_tokens = get_identifiers(code, 'java')
        prog_length = len(code_tokens)
        processed_code = " ".join(code_tokens)
        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_tgt)
        variable_names = statement

        adv_code = ''

        # 如果长度超了，就截断；这里的block_size是CodeBERT能接受的输入长度
        # 计算importance_score.
        importance_score, replace_token_positions, names_positions_dict = get_importance_score(self.args, self.bleu_file, idx, nl, code,
                                                                                               current_prob,
                                                                                               words, variable_names,
                                                                                               self.model_tgt,
                                                                                               self.tokenizer_tgt)

        if importance_score is None:
            return -1, None, None, None, None, None, None

        token_pos_to_score_pos = {}

        for i, token_pos in enumerate(replace_token_positions):
            token_pos_to_score_pos[token_pos] = i
        # 重新计算Importance score，将所有出现的位置加起来（而不是取平均）.
        names_to_importance_score = {}

        for name in names_positions_dict.keys():
            total_score = 0.0
            positions = names_positions_dict[name]
            for token_pos in positions:
                # 这个token在code中对应的位置
                # importance_score中的位置：token_pos_to_score_pos[token_pos]
                total_score += importance_score[token_pos_to_score_pos[token_pos]]

            names_to_importance_score[name] = total_score

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)
        # 根据importance_score进行排序

        final_words = copy.deepcopy(words)
        final_code = copy.deepcopy(code)
        nb_changed_var = 0  # 表示被修改的variable数量
        nb_changed_pos = 0
        is_success = -1
        replaced_words = {}
        query_times = 0
        for name_and_score in sorted_list_of_names:
            tgt_word = name_and_score[0]
            tgt_positions = names_positions_dict[tgt_word]

            all_substitues = []
            num = 0
            while num < 30:
                tmp_var = random.choice(self.idx2token)
                if isUID(tmp_var):
                    all_substitues.append(tmp_var)
                    num += 1

            # 得到了所有位置的substitue，并使用set来去重

            most_gap = 0.0
            candidate = None
            replace_examples = []

            substitute_list = []
            pred_list = []
            # 依次记录了被加进来的substitue
            # 即，每个temp_replace对应的substitue.

            query_times += len(all_substitues)
            for substitute in all_substitues:
                # temp_replace = copy.deepcopy(final_words)
                # for one_pos in tgt_positions:
                #     temp_replace[one_pos] = substitute

                substitute_list.append(substitute)
                # 记录了替换的顺序

                # 需要将几个位置都替换成sustitue_
                temp_code = get_example(final_code, tgt_word, substitute)

                new_example = get_new_example(idx, temp_code, nl)
                pred, _, _ = eval_bleu(self.args, new_example, self.model_tgt, self.tokenizer_tgt, self.bleu_file)
                pred_list.append(pred)
            for index, pred in enumerate(pred_list):
                # print("adv_bleu: {}, original_bleu: {}".format(pred[0], original_bleu))
                if pred[0] == 0.0:
                    is_success = 1
                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    candidate = substitute_list[index]
                    replaced_words[tgt_word] = candidate
                    adv_code = get_example(final_code, tgt_word, candidate)
                    print("%s SUC! %s => %s (%.5f => %.5f)" % \
                          ('>>', tgt_word, candidate,
                           current_prob, pred[0]), flush=True)
                    return is_success, adv_code
                else:
                    # 如果没有攻击成功，我们看probability的修改
                    gap = current_prob - pred[0]
                    # 并选择那个最大的gap.
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute_list[index]

            if most_gap > 0:

                nb_changed_var += 1
                nb_changed_pos += len(names_positions_dict[tgt_word])
                current_prob = current_prob - most_gap
                replaced_words[tgt_word] = candidate
                final_code = get_example(final_code, tgt_word, candidate)
                print("%s ACC! %s => %s (%.5f => %.5f)" % \
                      ('>>', tgt_word, candidate,
                       current_prob + most_gap,
                       current_prob), flush=True)
            else:
                replaced_words[tgt_word] = tgt_word

            adv_code = final_code

        return is_success, adv_code