from gensim.models.word2vec import Word2Vec
import pandas as pd
import os


def train_vocab(size):
    corpus=[]
    # f_code=open('../data1/JAVA/train/code.original_subtoken','r')
    f_code=open('../data1/JAVA/train/javadoc.original','r')
    print('read original code')
    for line in f_code:
        code=line.split(' ')
        corpus.append(code)
    f_code.close()
    print('read end')

    if not os.path.exists( '../Clone-detection/encoder/vocab/train1'):
        os.mkdir('../Clone-detection/vocab/train1')

    w2v = Word2Vec(corpus, vector_size=size, workers=16, sg=1, min_count=3)
    # w2v.save('../Clone-detection/vocab/train1/node_w2v_code_' + str(size))
    w2v.save('../Clone-detection/vocab/train1/node_w2v_summ_' + str(size))
    MAX_TOKENS = w2v.wv.vectors.shape[0]
    print('max:'+str(MAX_TOKENS))

if __name__=='__main__':
    print('start training word2vec:')
    train_vocab(64)
    print('training end!')