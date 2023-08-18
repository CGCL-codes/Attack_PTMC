import numpy as np
from replace_and_camelSplit import replace_a_to_b,split_c_and_s
# from get_bleu import return_bleu
from get_encoder import return_encoder
import bleu
import torch
import os
import logging
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)


from tree_sitter import Language, Parser

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
UNK='<unk>'


logger = logging.getLogger(__name__)

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


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 inputs,
                 labels,

    ):
        self.example_id = example_id
        self.inputs = inputs
        self.labels = labels
        self.attn_mask = np.array(labels) != 0
        self.loss_mask = np.array(labels) == 2


def read_examples(idx, code, nl):
    """Read examples from filename."""
    examples = []
    examples.append(
        Example(
            idx=idx,
            source=code,
            target=nl,
        )
    )
    return examples




def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        block_size = args.max_source_length + args.max_target_length
        code = tokenizer.encode(example.source)
        doc = tokenizer.encode(example.target)
        if stage == 'test':
            doc = []
        while len(doc) + 1 > args.max_target_length:
            doc = doc[:-1]
        while len(code) + 1 > args.max_source_length:
            code = code[:-1]
        if stage == 'train':
            inputs = code + [tokenizer.bos_token_id] + doc + [tokenizer.eos_token_id]
            labels = [1] * len(code) + [2] * (len(doc) + 1) + [0]
            assert len(inputs) <= block_size
            pad_len = block_size - len(inputs)
            inputs += [tokenizer.pad_token_id] * pad_len
            labels += [0] * pad_len
            assert len(inputs) == len(labels)
        else:
            inputs = code + [tokenizer.bos_token_id]
            labels = [1] * len(code) + [2]

        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source: {}".format(example.source))
                logger.info("target: {}".format(example.target))
                logger.info("inputs: {}".format(' '.join(map(str, inputs))))
                logger.info("labels: {}".format(' '.join(map(str, labels))))

        features.append(
            InputFeatures(
                example_index,
                inputs,
                labels,
            )
        )
    return features


class TextDataset(Dataset):
    def __init__(self, examples, args):
        self.examples = examples
        self.args = args

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item].inputs), torch.tensor(self.examples[item].labels), \
               torch.tensor(self.examples[item].attn_mask, dtype=torch.uint8), \
               torch.tensor(self.examples[item].loss_mask, dtype=torch.uint8)


def computer_word_saliency_cos(model,code,summary,variable_list,vocab,embeddings,max_token,
                               vocab_src,vocab_trg,max_token_src,max_token_trg):
    word_saliency_list=[]
    code_=split_c_and_s(code.split(' '))
    #_,encoder=return_bleu(code_,summary,model)

    encoder=return_encoder(code_,summary,model,vocab_src,vocab_trg,max_token_src,max_token_trg)

    for var in variable_list:
        var_index = vocab[var] if var in vocab else max_token
        embedding_var=embeddings[var_index]

        cos=cosine_similarity(embedding_var.reshape(1,64),encoder.reshape(1,64))[0][0]
        cos=(1.0+cos)*0.5
        word_saliency_list.append((var,cos))
    return word_saliency_list



def return_bleu(args, model, tokenizer, device, eval_examples,epoch=0):
    quert_cnt = 0
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
    eval_data = TextDataset(eval_features, args)

    # Calculate bleu
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    p = []
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels, attn_mask, loss_mask = batch
        with torch.no_grad():
            preds = model(inputs=inputs, labels=labels, attn_mask=attn_mask, loss_mask=loss_mask, pred=True)
            quert_cnt += 1
            # print("preds:", preds)
            # print("source_ids:", source_ids)
            # print("source_mask:", source_mask)
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                # print("t:", t)
                if 0 in t:
                    t = t[:t.index(0)]
                # print("t:", t)
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                # print("text:", text)
                p.append(text)
    model.train()
    predictions = []
    with open(os.path.join(args.output_dir, "test_{}.output".format(str(epoch))), 'w') as f, open(
            os.path.join(args.output_dir, "test_{}.gold".format(str(epoch))), 'w') as f1:
        for ref, gold in zip(p, eval_examples):

            predictions.append(str(gold.idx) + '\t' + ref)
            f.write(str(gold.idx) + '\t' + ref + '\n')
            f1.write(str(gold.idx) + '\t' + gold.target + '\n')

    (goldMap, predictionMap) = bleu.computeMaps(predictions,
                                                os.path.join(args.output_dir, "test_{}.gold".format(epoch)))
    dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    return dev_bleu, quert_cnt

def computer_best_substitution(model,code,summary,variable_list,nearest_k_dict,args, tokenizer, device, epoch, count):

    query_cnt = 0
    tmp_cnt = 0
    best_substitution_list=[]
    code_=split_c_and_s(code.split(' '))
    tmp_code = code
    #print(code_)
    print("old_bleu")
    # print("code_",code_)
    # print("summary",summary)
    summary = summary.replace("\n","")
    eval_examples = read_examples(count,code_,summary)
    old_bleu, tmp_cnt= return_bleu(args, model, tokenizer, device, eval_examples, epoch)
    query_cnt += tmp_cnt
    new_variable_list=[]
    # old_bleu,_=return_bleu(code_,summary,model)
    print("old_bleu: ",old_bleu)
    for var in variable_list:  # variable list  for one code
        new_variable_list.append(var)
        print("new_bleu")
        print("var", var)
        max_delta_bleu=0
        nearest_k=nearest_k_dict[var]   # a list
        best_new_var =nearest_k[0]

        for new_var in nearest_k:
            new_code_list=replace_a_to_b(code,var,new_var)
            new_code=split_c_and_s(new_code_list)

            print("new_var", new_var)
            eval_examples = read_examples(count, new_code, summary)
            new_bleu, tmp_cnt= return_bleu(args, model, tokenizer, device, eval_examples, epoch)
            query_cnt += tmp_cnt
            # new_bleu,_=return_bleu(new_code,summary,model)

            delta_bleu=old_bleu-new_bleu
            if max_delta_bleu< delta_bleu:
                max_delta_bleu=delta_bleu
                best_new_var=new_var
                if new_bleu == 0:
                    break

        best_substitution_list.append((var,best_new_var,max_delta_bleu))


        tmp_code_list = replace_a_to_b(tmp_code, var, best_new_var)
        tmp_code = split_c_and_s(tmp_code_list)

        eval_examples = read_examples(count, tmp_code, summary)
        tmp_bleu, tmp_cnt = return_bleu(args, model, tokenizer, device, eval_examples, epoch)
        query_cnt += tmp_cnt
        if tmp_bleu == 0:
            break

    return best_substitution_list, query_cnt, new_variable_list