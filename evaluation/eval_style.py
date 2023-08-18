import pandas as pd
import csv
from numpy import *
from collections import Counter
model_list = ["CodeBERT", "CodeGPT", "PLBART"]
task_list = ["Clone-detection", "Vulnerability-detection", "Code-summarization"]
for model in model_list:
    for task in task_list:
        adv_file = "/Dataset and Result/Result/{}/{}/result/attack_style_all.csv".format(
            model, task)
        if task == 'Code-summarization':
            attack_type = 'Attack Style'
        else:
            attack_type = 'Attack Type'
        print("adv_file:", adv_file)
        data = pd.read_csv(adv_file)  # 读取csv文件
        cnt_1000 = 0
        list_1000 = []
        query = data['Query Times'].values.tolist()
        time = data['Time Cost'].values.tolist()
        succ = data[attack_type]
        SUC = Counter(succ)
        print("-"*10 + model + "\t" + task + "-"*10)
        print("ASR: {:.2%}".format(SUC['style change'] / len(succ)))
        print("AMQ: {:.2f}".format(mean(query)))
        print("ART: {:.2f}".format(mean(time)))