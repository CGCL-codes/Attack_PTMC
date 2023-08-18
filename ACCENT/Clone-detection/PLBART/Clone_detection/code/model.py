# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.encoder_embed_dim*2, config.encoder_embed_dim)
        self.dropout = nn.Dropout(config.attention_dropout)
        self.out_proj = nn.Linear(config.encoder_embed_dim, 2)

    def forward(self, x, **kwargs):
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
    
        
    def forward(self, input_ids=None,prev_tokens_ids=None,lengths=None,labels=None): 
        assert lengths.size(-1) % 2 == 0
        input_ids=input_ids.view(-1,self.args.block_size)
        prev_tokens_ids=prev_tokens_ids.view(-1,self.args.block_size)
        lengths=lengths.view(-1,lengths.size(-1)//2)-1
        lengths=lengths.squeeze(-1)

        outputs=self.encoder(src_tokens=input_ids,src_lengths=lengths,prev_output_tokens=prev_tokens_ids,features_only=True)[0] # L * 2B * D
        outputs=outputs[range(input_ids.size(0)),lengths,:] # 2B * D
        logits=self.classifier(outputs)
        prob=F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob

    def get_results(self, dataset, batch_size, threshold=0.5):
        '''Given a dataset, return probabilities and labels.'''
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4,
                                     pin_memory=False)

        ## Evaluate Model

        eval_loss = 0.0
        self.eval()
        logits = []
        labels = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")
            prev_tokens_ids = batch[1].to("cuda")
            lengths = batch[2].to("cuda")
            label = batch[3].to("cuda")
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs,prev_tokens_ids,lengths, label)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())
        logits = np.concatenate(logits, 0)
        labels = np.concatenate(labels, 0)

        probs = logits
        pred_labels = [0 if first_softmax > threshold else 1 for first_softmax in logits[:, 0]]
        return probs, pred_labels

