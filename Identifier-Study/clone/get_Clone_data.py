import json
import random
import pandas as pd
from tqdm import tqdm


def get_code_pairs(file_path):
    url_to_code = {}
    with open('/'.join(file_path.split('/')[:-1]) + '/data.jsonl') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            url_to_code[js['idx']] = js['func']
    data = []
    cache = {}
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data.append((url1, url2, label, cache, url_to_code))
    code_pairs = []
    for sing_example in data:
        code_pairs.append([sing_example[0],
                           sing_example[1],
                           url_to_code[sing_example[0]],
                           url_to_code[sing_example[1]], label])
    return code_pairs

eval_data_file = "../../dataset/Clone-detection/test_sampled.txt"
source_codes = get_code_pairs(eval_data_file)
print(len(source_codes))

file = "data/data.csv"
df = pd.read_csv(file)

code_list = df["code"].tolist()
idx_list = df["idx"].tolist()

method_list = df["Method_statement"].tolist()
return_list = df["Return_statement"].tolist()
if_list = df["If_statement"].tolist()
throw_list = df["Throw_statement"].tolist()
try_list = df["Try_statement"].tolist()
for_list = df["For_statement"].tolist()
while_list = df["While_statement"].tolist()
switch_list = df["Switch_statement"].tolist()

method_cani = []
Return_cani = []
If_cani = []
Throw_cani = []
Try_cani = []
For_cani = []

for index in tqdm(range(len(source_codes))):
    code_pair = source_codes[index]
    label = code_pair[4]
    idx1 = int(code_pair[0])
    idx2 = int(code_pair[1])

    for i in range(len(code_list)):
        idx11 = idx_list[i]
        if idx1 == idx11:
            method_state = method_list[i].strip('[').strip(']').split(',')
            method_state = [] if method_state == [''] else method_state
            return_state = return_list[i].strip('[').strip(']').split(',')
            return_state = [] if return_state == [''] else return_state
            if_state = if_list[i].strip('[').strip(']').split(',')
            if_state = [] if if_state == [''] else if_state
            throw_state = throw_list[i].strip('[').strip(']').split(',')
            throw_state = [] if throw_state == [''] else throw_state
            try_state = try_list[i].strip('[').strip(']').split(',')
            try_state = [] if try_state == [''] else try_state
            for_state = for_list[i].strip('[').strip(']').split(',')
            for_state = [] if for_state == [''] else for_state
            while_state = while_list[i].strip('[').strip(']').split(',')
            while_state = [] if while_state == [''] else while_state
            switch_state = switch_list[i].strip('[').strip(']').split(',')
            switch_state = [] if switch_state == [''] else switch_state

            method_cani.append([idx1, idx2, label]) if len(method_state)>0 else method_cani
            Return_cani.append([idx1, idx2, label]) if len(return_state) > 0 else Return_cani
            If_cani.append([idx1, idx2, label]) if len(if_state) > 0 else If_cani
            Throw_cani.append([idx1, idx2, label]) if len(throw_state) > 0 else Throw_cani
            Try_cani.append([idx1, idx2, label]) if len(try_state) > 0 else Try_cani
            For_cani.append([idx1, idx2, label]) if len(for_state) > 0 else For_cani
            # if idx2 == idx_list[i]:
            #     return_state = return_list[i].strip('[').strip(']').split(',')
            #     return_state = [] if return_state == [''] else return_state
            #     if_state = if_list[i].strip('[').strip(']').split(',')
            #     if_state = [] if if_state == [''] else if_state
            #     throw_state = throw_list[i].strip('[').strip(']').split(',')
            #     throw_state = [] if throw_state == [''] else throw_state
            #     try_state = try_list[i].strip('[').strip(']').split(',')
            #     try_state = [] if try_state == [''] else try_state
            #     for_state = for_list[i].strip('[').strip(']').split(',')
            #     for_state = [] if for_state == [''] else for_state
            #     while_state = while_list[i].strip('[').strip(']').split(',')
            #     while_state = [] if while_state == [''] else while_state
            #     switch_state = switch_list[i].strip('[').strip(']').split(',')
            #     switch_state = [] if switch_state == [''] else switch_state
            #     if len(return_state)>0 and len(if_state)>0 and len(throw_state)>0 and len(try_state)>0 and len(for_state)>0:
            #         idx_2 = 1
            # if idx_1 == 1 and idx_2 ==1:
            #     sum += 1
            #     idx_1 = 0
            #     idx_2 = 0
            #     outfile.write('{}\t{}\t{}\n'.format(idx1, idx2, label))
print(len(method_cani),len(Return_cani), len(If_cani), len(Throw_cani), len(Try_cani), len(For_cani))

# method_cani = random.sample(method_cani, 1000)
# Return_cani = random.sample(Return_cani, 1000)
# If_cani = random.sample(If_cani, 1000)
# Throw_cani = random.sample(Throw_cani, 1000)
# Try_cani = random.sample(Try_cani, 1000)
# For_cani = random.sample(For_cani, 1000)
for state in ["method", "Return", "If", "Throw", "Try", "For"]:
    outfile_name = "data/{}_statement.txt".format(state)
    outfile = open(outfile_name, 'w')
    tmp_list = []
    if state == "method":
        tmp_list = method_cani
    elif state == "Return":
        tmp_list = Return_cani
    elif state == "If":
        tmp_list = If_cani
    elif state == "Throw":
        tmp_list = Throw_cani
    elif state == "Try":
        tmp_list = Try_cani
    elif state == "For":
        tmp_list = For_cani

    for line in tmp_list:
        outfile.write('{}\t{}\t{}\n'.format(line[0], line[1], line[2]))
    outfile.close()



