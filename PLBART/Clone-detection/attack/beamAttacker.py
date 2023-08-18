from utils import get_code_tokens
from utils import PLBARTCodeDataset, is_valid_identifier
import random
import copy
import pandas as pd
import time
import math
import torch
import operator
from run_parser import get_example
from run import InputFeatures, convert_examples_to_features

def get_statement_identifier(first_idx, identifiers):
    file_path = "../../../dataset/Clone-detection/data.csv"
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
    def __init__(self, args, model_tgt, tokenizer_tgt, tokenizer_mlm, model_mlm):
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.tokenizer_mlm = tokenizer_mlm
        self.model_mlm = model_mlm

    def is_vaild(self, code_token, identifier):
        if not is_valid_identifier(identifier):
            return False
        position = []
        for index, token in enumerate(code_token):
            if identifier == token:
                position.append(index)
        if all(x > self.args.block_size-2 for x in position):
            return False
        return True

    def perturb(self, example, code_1, code_2, all_substitues, tgt_word, iters, equal=False):
        is_success = -1
        final_code = copy.deepcopy(code_1)
        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        # print("attacker true label: ", orig_label)
        current_prob = max(orig_prob)

        candidate = None
        substitute_list = []
        all_substitues = list(set([subs.strip() for subs in all_substitues if subs != tgt_word]))
        cosine_list = []
        for sub in all_substitues:
            temp_code = get_example(code_1, tgt_word, sub)
            code1_tokens = [self.tokenizer_mlm.cls_token] + self.tokenizer_mlm.tokenize(code_1)[
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
        replace_examples = []

        for substitute in select_substitues:
            if not is_valid_identifier(substitute.strip()):
                continue
            substitute_list.append(substitute.strip())
            temp_replace = get_example(final_code, tgt_word, substitute.strip())
            new_feature = convert_examples_to_features(temp_replace,
                                                       code_2,
                                                       example[3].item(),
                                                       None, None,
                                                       self.tokenizer_tgt,
                                                       self.args, None)
            replace_examples.append(new_feature)
        new_dataset = PLBARTCodeDataset(replace_examples)
        logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
        for index, temp_prob in enumerate(logits):
            candidate = substitute_list[index]
            temp_label = preds[index]
            if temp_label != orig_label:
                # print("attacker predict label: ", temp_label)
                is_success = 1
                adv_code = get_example(final_code, tgt_word, candidate)
                return [[is_success, adv_code, candidate, temp_prob[orig_label]]]
            elif equal is True and temp_prob[temp_label] <= current_prob:
                gaps.append(
                    [is_success, get_example(final_code, tgt_word, candidate), candidate, temp_prob[temp_label]])
            elif equal is False and temp_prob[temp_label] < current_prob:
                gaps.append(
                    [is_success, get_example(final_code, tgt_word, candidate), candidate, temp_prob[temp_label]])

        if len(gaps) > 0:
            return gaps
        else:
            return []

    def beam_attack(self, orig_prob, example, substitutes, code_pair, statement_dict, beam_size):
        state_weight = {"For": 26.05, "Try": 23.89, "If": 23.37, "Method": 17.9, "Throw": 13.39, "Return": 11.26}
        first_probability = 26.05
        state_list = list(statement_dict.keys())
        label = example[3].item()
        result = {"succ": -1}
        code_1 = code_pair[2]
        code_2 = code_pair[3]

        # start beam attack
        iter = 0
        init_pop = {}
        final_pop = {}
        used_iden = []
        replace_info = ""
        tmp_code = code_1
        code_token = get_code_tokens(tmp_code)
        for key, identifiers in statement_dict.items():
            if iter == 0:
                used_iden += identifiers
                for identifier in identifiers:
                    if not self.is_vaild(code_token, identifier):
                        continue
                    gaps = self.perturb(example, code_1, code_2, substitutes[identifier], identifier, iter)
                    if len(gaps) > 0:
                        for gap in gaps:
                            is_success, final_code, candidate, current_prob = gap[0], gap[1], gap[2], gap[3]
                            if candidate is not None:
                                sequence = [iden for iden in identifiers if iden != identifier]
                                replace_info = identifier + ':' + candidate + ','
                                init_pop[replace_info] = {"adv_code": final_code, "prob": current_prob,
                                                          "original_var": [identifier],
                                                          "adv_var": [candidate], "sequence": sequence}
                                if is_success == 1:
                                    print("%s SUC! %s => %s (%.5f => %.5f)" % \
                                          ('>>', identifier, candidate,
                                           orig_prob,
                                           current_prob), flush=True)
                                    result["succ"] = 1
                                    result["adv_code"] = final_code
                                    result["replace_info"] = replace_info
                                    result["type"] = "Beam"
                                    return result
                    else:
                        init_pop["noChange"] = {"adv_code": code_1, "prob": orig_prob, "original_var": [],
                                                "adv_var": [], "sequence": identifiers}

                final_pop = dict(sorted(init_pop.items(), key=lambda x: x[1]['prob'])[:beam_size])

            num_iter = len(identifiers) - 1
            if iter > 0:
                tmp_pop = {}
                identifiers = [iden for iden in identifiers if iden not in used_iden]
                used_iden += identifiers
                final_pop_copy = copy.copy(final_pop)
                # if len(final_pop_copy) == 0:
                #     tmp_pop["noChange"] = {"adv_code": code_1, "prob": orig_prob, "original_var": [],
                #                                 "adv_var": [], "sequence": identifiers}
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
            # print("num_iter:", iter)
            for i_iter in range(num_iter):
                tmp_pop = {}
                final_pop_copy = copy.copy(final_pop)
                for replace_info, value in final_pop_copy.items():
                    if len(value["sequence"]) == 0:
                        continue
                    for seq in value["sequence"]:
                        if not self.is_vaild(code_token, seq):
                            continue
                        new_feature = convert_examples_to_features(value["adv_code"], code_2, example[3].item(), None, None,
                                                                   self.tokenizer_tgt, self.args, None)
                        new_example = PLBARTCodeDataset([new_feature])
                        gaps = self.perturb(new_example[0], value["adv_code"], code_2, substitutes[seq], seq, iter)
                        if len(gaps) > 0:
                            for gap in gaps:
                                is_success, final_code, candidate, current_prob = gap[0], gap[1], gap[2], gap[3]
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
                                               orig_prob,
                                               current_prob), flush=True)
                                        result["succ"] = 1
                                        result["adv_code"] = final_code
                                        result["replace_info"] = new_replace_info
                                        result["type"] = "Beam"
                                        return result
                        else:
                            tmp_pop[replace_info] = value

                select_dict = dict(list(tmp_pop.items()) + list(final_pop_copy.items()))
                final_pop = dict(sorted(select_dict.items(), key=lambda x: x[1]['prob'])[:beam_size])
                if operator.eq(list(final_pop.keys()), list(final_pop_copy.keys())):
                    break
                if i_iter != num_iter:
                    duplicate_key = [i for i in list(final_pop.keys()) if i in list(final_pop_copy.keys())]
                    if len(duplicate_key) > 0:
                        for pop_key in duplicate_key:
                            del final_pop[pop_key]
                # for replace_info, value in final_pop.items():
                #     print("iter select:", i_iter, replace_info, value["original_var"], value["sequence"], value["prob"])

            final_pop = final_pop
            iter += 1

        # final iter
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
        replace_dict = {}
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
                break
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
                    new_feature = convert_examples_to_features(value["adv_code"], code_2, label, None, None,
                                                               self.tokenizer_tgt, self.args, None)
                    example = PLBARTCodeDataset([new_feature])
                    gaps = self.perturb(example[0], value["adv_code"], code_2, replace_dict[seq], seq, iter,
                                        equal=True)
                    flag += len(gaps)
                    if len(gaps) > 0:
                        gap = gaps[0]
                        is_success, final_code, candidate, current_prob = gap[0], gap[1], gap[2], gap[3]
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
                                       orig_prob,
                                       0.0), flush=True)
                                result["succ"] = 1
                                result["adv_code"] = final_code
                                result["replace_info"] = new_replace_info
                                result["type"] = "Beam"
                                return result
                    else:
                        tmp_pop[replace_info] = value
            select_dict = dict(list(tmp_pop.items()) + list(final_pop_copy.items()))
            new_pop = dict(sorted(select_dict.items(), key=lambda x: x[1]['prob'])[:beam_size])
            if operator.eq(list(new_pop.keys()), list(final_pop_copy.keys())):
                break
            # for replace_info, value in new_pop.items():
            #     print("new select:", iter, i_iter, replace_info, value["original_var"], value["sequence"],
            #           value["prob"])
        return result
