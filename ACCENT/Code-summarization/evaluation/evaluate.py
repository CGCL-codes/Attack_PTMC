from attacker.evaluation.meteor.meteor import Meteor
from attacker.evaluation.rouge.rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

import numpy as np
import sys

result_file = '/home/zxq/code/zxq-gnn2/out/test_details_20200903_154515.txt'

f=open(result_file,'r')

gts={}
res={}
hyps=[]
refs=[]

count=0
for line in f:
    if count %2 ==0:
        refs.append(line)
    if count %2 ==1:
        hyps.append(line)
    count=count+1

print(len(refs))
print(len(hyps))
bleu_score=0
for hyp, ref in zip(hyps, refs):
    hyp = hyp.strip().split(' ')
    ref = ref.strip().split(' ')
    bleu_score += sentence_bleu([ref], hyp, smoothing_function = SmoothingFunction().method4)

print("score_Bleu: ")
print(bleu_score*1.0/len(hyps))

#score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
print("Meteor: ")
#print(score_Meteor)
#score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
print("ROUGe: ") 
#print(score_Rouge)

