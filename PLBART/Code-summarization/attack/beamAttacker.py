import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('../../../')
sys.path.append('../code')
sys.path.append('../../../python_parser')

from run_parser import get_identifiers, get_example
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from run import TextDataset, InputFeatures, convert_examples_to_features
from utils import CodeDataset, is_valid_identifier, get_code_tokens
from attacker import get_new_example
import random
import torch
import Levenshtein
import copy
import pandas as pd
import operator
import math
import bleu

def ids_to_strs(Y, sp):
    ids = []
    eos_id = sp.PieceToId("</s>")
    pad_id = sp.PieceToId("<pad>")
    for idx in Y:
        ids.append(int(idx))
        if int(idx) == eos_id or int(idx) == pad_id:
            break
    return sp.DecodeIds(ids)

def get_statement_identifier(first_idx, identifiers):
    file_path = "../../../dataset/Code-summarization/test.csv"
    df = pd.read_csv(file_path, encoding="utf-8")

    row = df.loc[df['idx'] == first_idx].squeeze()
    if row.empty:
        return {}

    statement_types = ["Method_statement", "Return_statement", "If_statement",
                       "Throw_statement", "Try_statement", "For_statement"]
    statement_dict = {}

    def clean_and_filter(statement):
        cleaned = statement[1:-1].replace(" ", "").split(",")
        return [var for var in cleaned if var in identifiers]

    for stmt_type in statement_types:
        filtered_identifiers = clean_and_filter(row[stmt_type])
        if filtered_identifiers:
            statement_dict[stmt_type.split('_')[0]] = filtered_identifiers

    # Identify the "Other" statements
    value_list = [i for p in statement_dict.values() for i in p]
    other_statements = [var for var in identifiers if var not in value_list]
    if other_statements:
        statement_dict["Other"] = other_statements

    return statement_dict

class Beam_Attacker(object):
    def __init__(self, args, model_tgt, tokenizer_tgt, tokenizer_mlm, model_mlm, bleu_file):
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.tokenizer_mlm = tokenizer_mlm
        self.model_mlm = model_mlm
        self.bleu_file = bleu_file
        self.query = 0

    def is_vaild(self, code_token, identifier):
        if not is_valid_identifier(identifier):
            return False
        position = []
        for index, token in enumerate(code_token):
            if identifier == token:
                position.append(index)
        if all(x > self.args.max_target_length-2 for x in position):
            return False
        return True

    def eval_bleu(self, example):
        self.query += 1
        bleu_file = self.bleu_file
        model = self.model_tgt
        tokenizer = self.tokenizer_tgt
        eval_features = convert_examples_to_features([example], tokenizer, self.args, stage='test')
        eval_data = TextDataset(eval_features, self.args)

        # Calculate bleu
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        model.eval()
        p = []
        for batch in eval_dataloader:
            batch = tuple(t.to(self.args.device) for t in batch)
            source_ids, target_ids, source_mask, target_mask = batch
            with torch.no_grad():
                preds = model(source_ids=source_ids, source_mask=source_mask)
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = ids_to_strs(t, tokenizer)
                    p.append(text)

        pre_summary = p[0]
        model.train()
        predictions = []
        if os.path.exists(bleu_file + "/dev.output"):
            os.remove(bleu_file + "/dev.output")
        if os.path.exists(bleu_file + "/dev.gold"):
            os.remove(bleu_file + "/dev.gold")
        with open((bleu_file + "/dev.output"), 'w') as f, open((bleu_file + "/dev.gold"), 'w') as f1:
            for ref, gold in zip(p, [example]):
                predictions.append(str(gold.idx) + '\t' + ref)
                f.write(str(gold.idx) + '\t' + ref + '\n')
                f1.write(str(gold.idx) + '\t' + gold.target + '\n')
        f.close()
        f1.close()
        try:
            (goldMap, predictionMap) = bleu.computeMaps(predictions, bleu_file + "/dev.gold")
            dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        except:
            dev_bleu = -1

        return dev_bleu, pre_summary, example.target

    def perturb(self, example, all_substitues, tgt_word, equal=False):
        is_success = -1
        code = example.source
        idx = example.idx
        nl = example.target
        current_prob, _, _ = self.eval_bleu(example)
        # print("current_prob: ", current_prob)
        final_code = copy.deepcopy(code)
        candidate = None
        substitute_list = []
        all_substitues = list(set([subs.strip() for subs in all_substitues if subs != tgt_word]))
        cosine_list = []
        for sub in all_substitues:
            temp_code = get_example(code, tgt_word, sub)
            code1_tokens = [self.tokenizer_mlm.cls_token] + self.tokenizer_mlm.tokenize(code)[
                                                            :self.args.block_size - 2] + [self.tokenizer_mlm.sep_token]
            code2_tokens = [self.tokenizer_mlm.cls_token] + self.tokenizer_mlm.tokenize(temp_code)[
                                                            :self.args.block_size - 2] + [self.tokenizer_mlm.sep_token]
            code1_ids = self.tokenizer_mlm.convert_tokens_to_ids(code1_tokens)
            code2_ids = self.tokenizer_mlm.convert_tokens_to_ids(code2_tokens)
            context_embeddings1 = self.model_mlm(torch.tensor(code1_ids)[None, :].to(self.args.device))[0]
            context_embeddings1 = context_embeddings1.reshape(context_embeddings1.size()[0],
                                                              context_embeddings1.size()[1] *
                                                              context_embeddings1.size()[2])
            context_embeddings2 = self.model_mlm(torch.tensor(code2_ids)[None, :].to(self.args.device))[0]
            context_embeddings2 = context_embeddings2.reshape(context_embeddings2.size()[0],
                                                              context_embeddings2.size()[1] *
                                                              context_embeddings2.size()[2])
            try:
                cosine_similarity = torch.cosine_similarity(context_embeddings1, context_embeddings2, dim=1).item()
                cosine_list.append(cosine_similarity)
            except:
                cosine_list.append(0)
        subs_dict = dict(zip(all_substitues, cosine_list))
        subs_dict = dict(sorted(subs_dict.items(), key=lambda x: x[1], reverse=True))
        select_substitues = list(subs_dict.keys())[:30]
        gaps = []
        for substitute in select_substitues:
            if not is_valid_identifier(substitute):
                continue
            substitute_list.append(substitute)
            temp_code = get_example(code, tgt_word, substitute)
            new_example = get_new_example(idx, temp_code, nl)
            bleu, pre_summary, ref_summary = self.eval_bleu(new_example[0])
            # print("ref_summary:", ref_summary)
            # print("pre_summary:", pre_summary)
            # print("per:", tgt_word, substitute, bleu)
            # print("-"*50)
            if bleu == 0.0:
                is_success = 1
                print("ref summary: ", ref_summary)
                print("adv summary: ", pre_summary)
                return [[is_success, temp_code, substitute, 0]]
            elif equal is True and bleu <= current_prob:
                gaps.append([is_success, temp_code, substitute, bleu])
            elif equal is False and bleu < current_prob:
                gaps.append([is_success, temp_code, substitute, bleu])

        if len(gaps) > 0:
            return gaps
        else:
            is_success = -2
            return []

    def beam_attack(self, original_bleu, example, substitutes, statement_dict, beam_size):
        state_weight = {"Method": 88.81, "For": 35.14, "If": 32.40, "Try": 30.77, "Return": 29.77, "Throw": 27.12}
        first_probability = 88.81
        state_list = list(statement_dict.keys())
        result = {"succ": -1}
        code = example.source
        idx = example.idx
        nl = example.target
        iter = 0
        init_pop = {}
        final_pop = {}
        used_iden = []
        replace_info = ""
        tmp_code = code
        code_token = get_code_tokens(tmp_code)
        replace_dict = {}
        for key, identifiers in statement_dict.items():
            if iter == 0:
                used_iden += identifiers
                for identifier in identifiers:
                    if not self.is_vaild(code_token, identifier):
                        continue
                    # is_success, final_code, candidate, current_prob = self.perturb(example, substitutes[identifier], identifier)
                    gaps = self.perturb(example, substitutes[identifier], identifier)
                    if len(gaps) > 0:
                        for gap in gaps:
                            is_success, final_code, candidate, current_prob = gap[0], gap[1], gap[2], gap[3]
                            # print(identifier, candidate, current_prob)
                            if candidate is not None:
                                sequence = [iden for iden in identifiers if iden != identifier]
                                replace_info = identifier + ':' + candidate + ','
                                init_pop[replace_info] = {"adv_code": final_code, "prob": current_prob,
                                                          "original_var": [identifier],
                                                          "adv_var": [candidate], "sequence": sequence}
                                if is_success == 1:
                                    print("%s SUC! %s => %s (%.5f => %.5f)" % \
                                          ('>>', identifier, candidate,
                                           original_bleu,
                                           0.0), flush=True)
                                    result["succ"] = 1
                                    result["adv_code"] = final_code
                                    result["replace_info"] = replace_info
                                    result["type"] = "Beam"
                                    return result
                    else:
                        init_pop["noChange"] = {"adv_code": tmp_code, "prob": original_bleu, "original_var": [],
                                                "adv_var": [], "sequence": identifiers}

                sort_dit = dict(sorted(init_pop.items(), key=lambda x: x[1]['prob']))
                final_pop = {k: sort_dit[k] for k in list(sort_dit)[:beam_size]}

            num_iter = len(identifiers) - 1
            if iter > 0:
                tmp_pop = {}
                identifiers = [iden for iden in identifiers if iden not in used_iden]
                used_iden += identifiers
                final_pop_copy = copy.copy(final_pop)
                if len(final_pop_copy) == 0:
                    tmp_pop["noChange"] = {"adv_code": tmp_code, "prob": original_bleu, "original_var": [],
                                           "adv_var": [], "sequence": identifiers}

                for replace_info, value in final_pop_copy.items():
                    tmp_pop[replace_info] = {"adv_code": value["adv_code"], "prob": value["prob"],
                                             "original_var": value["original_var"],
                                             "adv_var": value["adv_var"], "sequence": identifiers}
                final_pop = tmp_pop
                state = state_list[iter]
                if state in state_weight:
                    probability = state_weight[state]
                    num_iter = math.ceil(len(identifiers) * probability / first_probability)
                else:
                    probability = state_weight.get(list(state_weight.keys())[-1])
                    num_iter = math.ceil(len(identifiers) * probability / first_probability)
            # for replace_info, value in final_pop.items():
            #     print("----", iter, replace_info, value["original_var"], value["sequence"], value["prob"])
            # print("num_iter:", num_iter)
            for i_iter in range(num_iter):
                tmp_pop = {}
                final_pop_copy = copy.copy(final_pop)
                for replace_info, value in final_pop_copy.items():
                    if len(value["sequence"]) == 0:
                        continue
                    for seq in value["sequence"]:
                        if not self.is_vaild(code_token, seq):
                            continue
                        new_example = get_new_example(idx, value["adv_code"], nl)
                        # is_success, final_code, candidate, current_prob = self.perturb(new_example[0], substitutes[seq], seq)
                        gaps = self.perturb(new_example[0], substitutes[seq], seq)
                        if len(gaps) > 0:
                            for gap in gaps:
                                is_success, final_code, candidate, current_prob = gap[0], gap[1], gap[2], gap[3]
                                # print(seq, candidate, current_prob)
                                if candidate is not None:
                                    original_var = value["original_var"] + [seq]
                                    adv_var = value["adv_var"] + [candidate]
                                    new_replace_info = ''
                                    for info_i in range(len(original_var)):
                                        new_replace_info += original_var[info_i] + ':' + adv_var[info_i] + ','
                                    sequence = [iden for iden in value["sequence"] if iden not in original_var]
                                    tmp_pop[new_replace_info] = {"adv_code": final_code, "prob": current_prob,
                                                                 "original_var": original_var,
                                                                 "adv_var": adv_var, "sequence": sequence}
                                    if is_success == 1:
                                        print("%s SUC! %s => %s (%.5f => %.5f)" % \
                                              ('>>', original_var, adv_var,
                                               original_bleu,
                                               0.0), flush=True)
                                        result["succ"] = 1
                                        result["adv_code"] = final_code
                                        result["replace_info"] = new_replace_info
                                        result["type"] = "Beam"
                                        return result
                        else:
                            tmp_pop[replace_info] = value

                select_dict = dict(list(tmp_pop.items()) + list(final_pop_copy.items()))
                sort_dit = dict(sorted(select_dict.items(), key=lambda x: x[1]['prob']))
                final_pop = {k: sort_dit[k] for k in list(sort_dit)[:beam_size]}
                if operator.eq(list(final_pop.keys()), list(final_pop_copy.keys())):
                    break
            final_pop = final_pop
            iter += 1

        max_len = 0
        for replace_info, value in final_pop.items():
            if len(value["original_var"]) > max_len:
                final_pop = {replace_info: value}
                max_len = len(value["original_var"])

        replace_identifier = []
        adv_identifier = []
        for replace_info, value in final_pop.items():
            # print("***", replace_info, value["original_var"], value["sequence"], value["prob"])
            replace_identifier += value["original_var"]
            adv_identifier += value["adv_var"]
        for identifier, adv in zip(replace_identifier, adv_identifier):
            if adv not in list(replace_dict.keys()):
                subs = substitutes[identifier]
                subs = list(set([sub.strip() for sub in subs]))
                subs.remove(adv)
                subs.append(identifier)
                replace_dict[adv] = subs
        new_pop = {}
        for replace_info, value in final_pop.items():
            new_pop[replace_info] = {"adv_code": value["adv_code"], "prob": value["prob"],
                                     "original_var": value["original_var"],
                                     "adv_var": value["adv_var"], "sequence": value["adv_var"]}
        flag = 0
        for i_iter in range(len(adv_identifier)):
            if i_iter > 0 and flag == 0:
                continue
            tmp_pop = {}
            final_pop_copy = copy.copy(new_pop)
            for replace_info, value in final_pop_copy.items():
                if len(value["sequence"]) == 0:
                    continue
                for seq in value["sequence"]:
                    try:
                        code_token = get_code_tokens(value["adv_code"])
                    except:
                        print("syntax errors!")
                        continue
                    if not self.is_vaild(code_token, seq):
                        continue
                    new_example = get_new_example(idx, value["adv_code"], nl)
                    gaps = self.perturb(new_example[0], replace_dict[seq], seq, equal=True)
                    flag += len(gaps)
                    if len(gaps) > 0:
                        gap = gaps[0]
                        is_success, final_code, candidate, current_prob = gap[0], gap[1], gap[2], gap[3]
                        # print(seq, candidate, current_prob)
                        if candidate is not None:
                            original_var, adv_var = [], []
                            if candidate in value["original_var"]:
                                value["original_var"].remove(candidate)
                                value["adv_var"].remove(seq)
                                original_var = value["original_var"]
                                adv_var = [candidate if i == seq else i for i in value["adv_var"]]
                            else:
                                original_var = value["original_var"]
                                adv_var = [candidate if i == seq else i for i in value["adv_var"]]
                            new_replace_info = ''
                            for info_i in range(len(original_var)):
                                new_replace_info += original_var[info_i] + ':' + adv_var[info_i] + ','
                            sequence = [iden for iden in value["sequence"] if iden not in adv_var]
                            tmp_pop[new_replace_info] = {"adv_code": final_code, "prob": current_prob,
                                                         "original_var": original_var,
                                                         "adv_var": adv_var, "sequence": sequence}
                            if is_success == 1:
                                print("%s SUC in Final! %s => %s (%.5f => %.5f)" % \
                                      ('>>', original_var, adv_var,
                                       original_bleu,
                                       0.0), flush=True)
                                result["succ"] = 1
                                result["adv_code"] = final_code
                                result["replace_info"] = new_replace_info
                                result["type"] = "Beam"
                                return result
                    else:
                        tmp_pop[replace_info] = value

            select_dict = dict(list(tmp_pop.items()) + list(final_pop_copy.items()))
            sort_dit = dict(sorted(select_dict.items(), key=lambda x: x[1]['prob']))
            new_pop = {k: sort_dit[k] for k in list(sort_dit)[:beam_size]}
            if operator.eq(list(new_pop.keys()), list(final_pop_copy.keys())):
                break
            # for replace_info, value in new_pop.items():
            #     print("new select:", iter, i_iter, replace_info, value["original_var"], value["sequence"],
            #           value["prob"])

        return result
