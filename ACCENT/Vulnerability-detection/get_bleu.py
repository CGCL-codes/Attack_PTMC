from c2nl.inputters.utils import process_examples

import c2nl.inputters.dataset as dataset
import c2nl.inputters.vector as vector
from main.train import eval_accuracies
import os
import torch
import torch.nn as nn

def pooling(x):
    x=x.view(1,1,-1)
    
    out=nn.MaxPool1d(8,stride=8)
    x=out(x)
    x=x.squeeze()
    return x

def tensor_to_numpy(x):
    return x.numpy()



def validate_official(data_loader, model):

    # Run through examples
    examples = 0
    sources, hypotheses, references, copy_dict = dict(), dict(), dict(), dict()
    with torch.no_grad():

        for idx,ex in enumerate(data_loader):

            batch_size = ex['batch_size']
            ex_ids = list(range(idx * batch_size, (idx * batch_size) + batch_size))
            predictions, targets, copy_info,encoder = model.predict(ex, replace_unk=True)
            print("predictions:",predictions)
            print("targets:",targets)
            print("copy_info:",copy_info)
            print("encoder:",encoder)

            src_sequences = [code for code in ex['code_text']]
            examples += batch_size
            for key, src, pred, tgt in zip(ex_ids, src_sequences, predictions, targets):
                hypotheses[key] = [pred]
                references[key] = tgt if isinstance(tgt, list) else [tgt]
                sources[key] = src

            if copy_info is not None:
                copy_info = copy_info.cpu().numpy().astype(int).tolist()
                for key, cp in zip(ex_ids, copy_info):
                    copy_dict[key] = cp

    copy_dict = None if len(copy_dict) == 0 else copy_dict
    bleu, rouge_l, meteor, precision, recall, f1 = eval_accuracies(hypotheses,
                                                                   references,
                                                                   copy_dict,
                                                                   sources=sources,
                                                                   filename=None,
                                                                   print_copy_info=False,mode='dev')

    result = dict()
    result['bleu'] = bleu
    result['rouge_l'] = rouge_l
    result['meteor'] = meteor
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1
    result['encoder']=encoder

    return result


'''
java:0 python:1 LANG_ID 
SOURCE_TAG=None
java:150   python:400 MAX_SRC_LEN
java:50  python:30  MAX_TGT_LEN
CODE_TAG_TYPE='subtoken'
'''

LANG_ID=1   #1
SOURCE_TAG=None
MAX_SRC_LEN=400 #150
MAX_TGT_LEN=30 #50
CODE_TAG_TYPE='subtoken'



def return_bleu(code,summary,model):
    lang_id=LANG_ID
    source=code
    source_tag=SOURCE_TAG
    target=summary
    max_src_len=MAX_SRC_LEN
    max_tgt_len=MAX_TGT_LEN
    code_tag_type=CODE_TAG_TYPE
    uncase = True
    test_split = False

    exs=[process_examples(lang_id,source,source_tag,target,max_src_len,max_tgt_len,code_tag_type,uncase,test_split)]

    exs_dataset=dataset.CommentDataset(exs,model)

    exs_sampler = torch.utils.data.sampler.RandomSampler(exs_dataset)

    exs_loader = torch.utils.data.DataLoader(
        exs_dataset,
        batch_size=1,
        sampler=exs_sampler,
        num_workers=1,
        collate_fn=vector.batchify,  ##是否合并样本列表以形成一小批Tensor
        pin_memory=True, # (args.cuda)
        drop_last=False
    )

    print("code:",code)
    print("summary:",summary)
    result=validate_official(exs_loader,model)

    bleu_sorce=result['bleu']
    memory_bank = result['encoder'].squeeze(0)
    memory_bank_ = pooling((torch.max(memory_bank, 0)[0].data))  
 
    memory_bank_=tensor_to_numpy(memory_bank_)

    return bleu_sorce,memory_bank_