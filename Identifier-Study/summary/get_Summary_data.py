import json
import random
import pandas as pd
import csv
from tqdm import tqdm

test_idxs = []
with open("../../dataset/Code-summarization/test_sampled.jsonl") as f:
    for line in f:
        line = line.strip()
        js = json.loads(line)
        test_idxs.append(js["idx"])
print(len(test_idxs), test_idxs[0])

file = "data/test.csv"
df = pd.read_csv(file)

code_list = df["code"].tolist()
summary_list = df["summary"].tolist()

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

print(len(code_list))
for i in range(len(code_list)):
    if i in test_idxs:
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

        method_cani.append([i, code_list[i], summary_list[i], method_state]) if len(method_state)>0 else method_cani
        Return_cani.append([i, code_list[i], summary_list[i], return_state]) if len(return_state) > 0 else Return_cani
        If_cani.append([i, code_list[i], summary_list[i], if_state]) if len(if_state) > 0 else If_cani
        Throw_cani.append([i, code_list[i], summary_list[i], throw_state]) if len(throw_state) > 0 else Throw_cani
        Try_cani.append([i, code_list[i], summary_list[i], try_state]) if len(try_state) > 0 else Try_cani
        For_cani.append([i, code_list[i], summary_list[i], for_state]) if len(for_state) > 0 else For_cani

print(len(method_cani),len(Return_cani), len(If_cani), len(Throw_cani), len(Try_cani), len(For_cani))

for state in ["Method", "Return", "If", "Throw", "Try", "For"]:
    outfile_name = "data/{}_statement.csv".format(state)
    outfile = open(outfile_name, 'w', encoding='utf-8')
    writer = csv.writer(outfile)
    writer.writerow(["idx", "code", "summary", "identifier"])
    tmp_list = []
    if state == "Method":
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

    writer.writerows(tmp_list)
    outfile.close()



