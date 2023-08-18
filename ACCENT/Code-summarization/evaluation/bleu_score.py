
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

#refs comm ===> example: 'xxxxxxxxxxxxxxx', 'xxxxxxxxxxx'

def com_bleu(ref,gene):
    ref_=[ref.split(' ')]
    gene_=gene.split(' ')
    print(ref_)
    print(gene_)
    bleu=sentence_bleu(ref_, gene_, smoothing_function=SmoothingFunction().method4)
    return bleu

#b=com_bleu('a b cdf dfdg','a b cdf dfdg')
#print(b)
