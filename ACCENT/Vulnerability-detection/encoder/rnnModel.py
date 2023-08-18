import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import time
from configuation import Configuration


# Encoder
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, src_vocab, emb_dim, hid_dim, n_layers, dropout):
        '''
        :param src_vocab_size: 输入源词库的大小
        :param src_vocab: numpy输入词嵌入[src_vocab_size, emb_dim]
        :param emb_dim:  输入单词Embedding的维度
        :param hid_dim: 隐层的维度
        :param n_layers: 几个隐层
        :param dropout:  dropout参数 0.5
        '''

        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(src_vocab_size, emb_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(src_vocab))
        self.embedding.weight.requires_grad = False

        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers = n_layers, dropout=dropout)


    def forward(self, src):
        # src = [src sent len, batch size] 这句话的长度和batch大小

        embedded = self.embedding(src)



        # embedded = [src sent len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)   # n_layers * seq_len * embed_dim

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, target_vocab, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.target_vocab_size = target_vocab_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(target_vocab_size, emb_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(target_vocab))
        self.embedding.weight.requires_grad = False

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.out = nn.Linear(hid_dim, target_vocab_size)



    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.embedding(input)

        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # !! sent len and n directions will always be 1 in the decoder,therefore:

        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell

# seq2seq
class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, target_vocab):
        """
        :param src_vocab: numpy输入词嵌入[src_vocab_size, emb_dim]
        :param target_vocab: numpy输入词嵌入[target_vocab_size, emb_dim]
        """
        super().__init__()
        self.config = Configuration()
        self.encoder = Encoder(self.config.src_vocab_size, src_vocab, self.config.emb_dim, self.config.hid_dim, self.config.n_layers, self.config.dropout)
        self.decoder = Decoder(self.config.target_vocab_size, target_vocab, self.config.emb_dim, self.config.hid_dim, self.config.n_layers, self.config.dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.config.target_vocab_size

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size,
                              trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder

        hidden, cell = self.encoder.forward(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, max_len):
            # insert input token embedding, previous hidden and previous cell states

            # receive output tensor (predictions) and new hidden and cell states

            output, hidden, cell = self.decoder.forward(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token

            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, 203 use predicted token
            # 在 模型训练速度 和 训练测试差别不要太大 作一个均衡
            input = trg[t] if teacher_force else top1

        return outputs, hidden