import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

sys.path.append('../')
sys.path.append('../python_parser')

import pandas as pd
import javalang
import torch
# from run_parser import get_identifiers
# from utils import get_replaced_code, get_code_tokens
import json
from tqdm import tqdm
from numpy import *
import Levenshtein
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForMaskedLM, AutoTokenizer

def get_code_tokens(code):
    tokens = javalang.tokenizer.tokenize(code)
    code_tokens = [token.value for token in tokens]
    return code_tokens

device = torch.device("cuda")

model_list = ["CodeBERT", "CodeGPT", "PLBART"]
task_list = ["Clone-detection", "Vulnerability-detection", "Code-summarization"]
# task_list = ["Code-summarization"]
attack_list = ["mhm", "alert", "wir", "accent"]
for model in model_list:
    for task in task_list:
        for attack_method in attack_list:
            adv_file = "../{}/{}/attack/result/attack_{}_all.csv".format(model, task, attack_method)
# adv_file = "../PLBART/Vulnerability-detection2/attack/result/attack_alert_all.csv"
# adv_file = "../PLBART/Clone-detection/attack/result/attack_beam.csv"
# adv_file = "../CodeBERT/Code-summarization/attack/result/attack_beam.csv"
# adv_file = "../CodeGPT/Vulnerability-detection2/attack/result/attack_beam.csv"

            attack = adv_file.split(".")[-2].split("_")[-1]
            df = pd.read_csv(adv_file, encoding="utf-8")
            code_list = df["Original Code"]
            adv_code_list = df["Adversarial Code"]
            replaced_list = df["Replaced Identifiers"]
            query_list = df["Query Times"]
            time_list = df["Time Cost"]
            total_identifiers_list = df["Identifier Num"]
            total_token_list = df["Program Length"]
            type_list = df["Type"]
            skip_var_num = 0
            skip_length_num = 0
            succ_num = 0
            # print("total: ", len(code_list))
            model_mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
            tokenizer_mlm = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
            model_mlm.to(device)
            cosine_list = []
            edit_list = []
            replece_var_list = []
            replece_token_list = []

            succ_var_list = []
            succ_token_list = []
            succ_query_list = []
            succ_time_list = []
            for index, type in enumerate(type_list):

                if type != '0':
                    succ_query_list.append(query_list[index])
                    succ_time_list.append(time_list[index])
                    orginal_code = code_list[index]
                    adv_code = adv_code_list[index]
                    code1_tokens = get_code_tokens(orginal_code)
                    # _, code1_tokens = get_identifiers(orginal_code,"java")
                    try:
                        code2_tokens = get_code_tokens(adv_code)
                        # _, code2_tokens = get_identifiers(adv_code, "java")
                        code2_tokens = ['"\\n"' if i == '"\n"' else i for i in code2_tokens]
                    except Exception as e:
                        # print(e)
                        # print(orginal_code)
                        # print(adv_code)
                        skip_var_num += 1
                        continue
                    # print(code1_tokens)
                    # print(code2_tokens)
                    if not len(code1_tokens) == len(code2_tokens):
                        # print(index, len(code1_tokens), len(code2_tokens))
                        # print(code1_tokens)
                        # print(code2_tokens)
                        skip_length_num += 1
                    if len(code1_tokens) == len(code2_tokens):
                        succ_num += 1
                        code1_tokens = [tokenizer_mlm.cls_token] + code1_tokens[:510] + [tokenizer_mlm.sep_token]
                        code2_tokens = [tokenizer_mlm.cls_token] + code2_tokens[:510] + [tokenizer_mlm.sep_token]
                        code1_ids = tokenizer_mlm.convert_tokens_to_ids(code1_tokens)
                        code2_ids = tokenizer_mlm.convert_tokens_to_ids(code2_tokens)
                        context_embeddings1 = model_mlm(torch.tensor(code1_ids)[None, :].to(device))[0]
                        context_embeddings1 = context_embeddings1.reshape(context_embeddings1.size()[0],
                                                                          context_embeddings1.size()[1] *
                                                                          context_embeddings1.size()[2])
                        context_embeddings2 = model_mlm(torch.tensor(code2_ids)[None, :].to(device))[0]
                        context_embeddings2 = context_embeddings2.reshape(context_embeddings2.size()[0],
                                                                          context_embeddings2.size()[1] *
                                                                          context_embeddings2.size()[2])
                        cosine_similarity = torch.cosine_similarity(context_embeddings1, context_embeddings2, dim=1).item()
                        cosine_list.append(cosine_similarity)
                        # Compute edit distance
                        total_edit_distance = 0
                        for i in range(len(code1_tokens)):
                            edit_distance = Levenshtein.distance(code1_tokens[i], code2_tokens[i])
                            total_edit_distance += edit_distance
                        edit_list.append(total_edit_distance)

                        replaced_info = replaced_list[index]
                        replaced_info = replaced_info.split(",")[:-1]
                        replace_iden = []
                        for replace in replaced_info:
                            replace_pair = replace.split(":")
                            if replace_pair[0] != replace_pair[1]:
                                replace_iden.append(replace_pair[0])
                        replece_var_list.append(len(replace_iden))
                        #
                        sum_token = 0
                        for tok in code1_tokens:
                            if tok in replace_iden:
                                sum_token += 1
                        replece_token_list.append(sum_token)

            # print("Skip var: {}. length: {}".format(skip_var_num, skip_length_num))
            # print("Total: ", succ_num)
            print("*"*30)
            print("{}-{}-{}".format(model, task, attack_method))
            print("ASR: {:.2%}".format(succ_num/len(code_list)))
            print("AMQ: {:.2f}".format(mean(succ_query_list)))
            print("ART: {:.2f}".format(mean(succ_time_list)))
            print("ICR: {:.2%}".format(sum(replece_var_list) / sum(total_identifiers_list)))
            print("TCR: {:.2%}".format(sum(replece_token_list) / sum(total_token_list)))
            print("ACS: {:.4f}".format(mean(cosine_list)))
            print("AED: {:.2f}".format(mean(edit_list)))
