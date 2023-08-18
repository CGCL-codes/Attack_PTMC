import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import sys
import copy
import torch
import pandas as pd
from tqdm import tqdm

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../python_parser')

# from attacker import
from python_parser.run_parser import get_identifiers, remove_comments_and_docstrings
from utils import _tokenize, get_identifier_posistions_from_code, get_masked_code_by_position, \
    get_substitues, is_valid_identifier
from transformers import (RobertaForMaskedLM, RobertaTokenizer, pipeline)


def main():

    eval_data = []
    base_model = "microsoft/codebert-base-mlm"
    block_size = 512
    codebert_mlm = RobertaForMaskedLM.from_pretrained(base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(base_model)
    codebert_mlm.to('cuda')
    fill_mask = pipeline('fill-mask', model=codebert_mlm, tokenizer=tokenizer_mlm, device=0)

    url_to_code = {}
    index = 0
    with open('../../dataset/Vulnerability-detection2/test.jsonl') as f:
        for line in f:
            item = {}
            line = line.strip()
            js = json.loads(line)
            item["idx"] = js["idx"]
            item["func"] = js["func"]
            eval_data.append(item)
            index += 1
    file = "data/test.csv"
    df = pd.read_csv(file, encoding="utf-8")
    code_list = df["code"].tolist()
    idx_list = df["idx"].tolist()
    All_list = df["All_identifier"].tolist()

    eval_data1 = eval_data[0:1000]
    eval_data2 = eval_data[1000:2000]
    eval_data3 = eval_data[2000:3000]
    eval_data4 = eval_data[3000:len(eval_data)]

    store_path = "data/data_subs_test4.jsonl"
    with open(store_path, "w") as wf:
        for item in tqdm(eval_data4):
            # print("item:", item["idx"])
            identifiers = ""
            try:
                idens, code_tokens = get_identifiers(remove_comments_and_docstrings(item["func"], "java"),
                                                           "java")
            except:
                idens, code_tokens = get_identifiers(item["func"], "java")
            processed_code = " ".join(code_tokens)

            words, sub_words, keys = _tokenize(processed_code, tokenizer_mlm)

            variable_names = []
            for name in idens:
                if ' ' in name[0].strip():
                    continue
                variable_names.append(name[0])
            # print(variable_names)

            for index in range(len(idx_list)):
                if int(item["idx"]) == idx_list[index]:
                    identifiers = All_list[index].replace(" ","").strip('[').strip(']').split(',')
                    identifiers = [] if identifiers == [''] else identifiers
            # print("iden:",iden)
            # print(type(variable_names), type(identifiers), variable_names, identifiers)
            if len(variable_names) == 0 and len(identifiers) == 0:
                continue
            elif len(variable_names) == 0:
                variable_names = identifiers
            elif len(identifiers) == 0:
                variable_names = variable_names
            else:
                variable_names = list(set(variable_names+identifiers))
            # print(variable_names)
            item["identifiers"] = variable_names

            sub_words = [tokenizer_mlm.cls_token] + sub_words[:block_size - 2] + [tokenizer_mlm.sep_token]

            input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])

            word_predictions = codebert_mlm(input_ids_.to('cuda'))[0].squeeze()  # seq-len(sub) vocab
            word_pred_scores_all, word_predictions = torch.topk(word_predictions, 60, -1)  # seq-len k

            word_predictions = word_predictions[1:len(sub_words) + 1, :]
            word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]

            names_positions_dict = get_identifier_posistions_from_code(words, variable_names)

            variable_substitue_dict = {}

            with torch.no_grad():
                orig_embeddings = codebert_mlm.roberta(input_ids_.to('cuda'))[0]
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            for tgt_word in names_positions_dict.keys():
                tgt_positions = names_positions_dict[tgt_word]  # the positions of tgt_word in code

                # alert
                all_substitues_alert = []
                for one_pos in tgt_positions:
                    if keys[one_pos][0] >= word_predictions.size()[0]:
                        continue
                    substitutes = word_predictions[keys[one_pos][0]:keys[one_pos][1]]  # L, k
                    word_pred_scores = word_pred_scores_all[keys[one_pos][0]:keys[one_pos][1]]

                    orig_word_embed = orig_embeddings[0][keys[one_pos][0] + 1:keys[one_pos][1] + 1]

                    similar_substitutes = []
                    similar_word_pred_scores = []
                    sims = []
                    subwords_leng, nums_candis = substitutes.size()

                    for i in range(nums_candis):
                        new_ids_ = copy.deepcopy(input_ids_)
                        new_ids_[0][keys[one_pos][0] + 1:keys[one_pos][1] + 1] = substitutes[:, i]

                        with torch.no_grad():
                            new_embeddings = codebert_mlm.roberta(new_ids_.to('cuda'))[0]
                        new_word_embed = new_embeddings[0][keys[one_pos][0] + 1:keys[one_pos][1] + 1]

                        sims.append((i, sum(cos(orig_word_embed, new_word_embed)) / subwords_leng))

                    sims = sorted(sims, key=lambda x: x[1], reverse=True)

                    for i in range(int(nums_candis / 2)):
                        similar_substitutes.append(substitutes[:, sims[i][0]].reshape(subwords_leng, -1))
                        similar_word_pred_scores.append(word_pred_scores[:, sims[i][0]].reshape(subwords_leng, -1))

                    similar_substitutes = torch.cat(similar_substitutes, 1)
                    similar_word_pred_scores = torch.cat(similar_word_pred_scores, 1)

                    substitutes = get_substitues(similar_substitutes,
                                                 tokenizer_mlm,
                                                 codebert_mlm,
                                                 1,
                                                 similar_word_pred_scores,
                                                 0)
                    all_substitues_alert += substitutes

                # fill-mask
                tgt_word_dict = {tgt_word: names_positions_dict[tgt_word]}
                # print(tgt_word_dict)
                masked_token_list, replace_token_positions = get_masked_code_by_position(words, tgt_word_dict)
                all_substitues_mask = []
                for masked_token in masked_token_list:
                    n = int(len(masked_token) / 200)
                    for i in range(n):
                        masked_token_part = masked_token[i * 200: (i + 1) * 200]
                        if "<mask>" in masked_token_part and masked_token_part[0] != "<mask>" and masked_token_part[-1] != "<mask>":
                            code = " ".join(masked_token_part)
                            try:
                                outputs = fill_mask(code)
                            except:
                                continue
                            subs_list = [subs_dict["token_str"] for subs_dict in outputs]
                            for subs in subs_list:
                                subs = subs.strip()
                                if not subs == tgt_word:
                                    all_substitues_mask.append(subs)

                all_substitues = all_substitues_alert + all_substitues_mask
                all_substitues = set(all_substitues)

                for tmp_substitue in all_substitues:
                    if tmp_substitue.strip() in variable_names:
                        continue
                    if not is_valid_identifier(tmp_substitue.strip()):
                        continue
                    try:
                        variable_substitue_dict[tgt_word].append(tmp_substitue.strip())
                    except:
                        variable_substitue_dict[tgt_word] = [tmp_substitue.strip()]
            item["substitutes"] = variable_substitue_dict
            wf.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    main()