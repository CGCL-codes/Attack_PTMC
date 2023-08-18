import pickle
import json
import pandas as pd
import numpy as np

# filename = "var_for_allCode_test"
filename = "var_for_everyCode_test11"
totalnum = 3942
f = open(filename+'.pkl','rb')
#使用load的方法将数据从pkl文件中读取出来
data = pickle.load(f)
print(type(data))
print(data.keys())
# <class 'pandas.core.frame.DataFrame'>
# Index(['id', 'all vars'], dtype='object')
# Index(['id', 'variable'], dtype='object')
totalcnt=0
with open(filename+".txt", 'w', encoding='utf-8') as f:
    for i in range(totalnum):
        mydata1 = data.loc[i, 'id' ]
        mydata2 = data.loc[i, 'variable' ]
        totalcnt += len(mydata2)
        # f.write("---------1----------")
        f.write(str(mydata1)+'\t'+str(mydata2)+'\n')
    f.write("totalcnt: "+str(totalcnt))
# #     f.write("---------2----------")
# #     f.write(str(data['labels']))
# #     f.write("---------3----------")
# #     f.write(str(data['uids']))
# #     f.write("---------4----------")
# #     f.write(str(data['raw_te']))
# #     f.write("---------5----------")
# #     f.write(str(data['y_te']))
# #     f.write("---------6----------")
# #     f.write(str(data['x_te']))
# #     f.write("---------7----------")
# #     f.write(str(data['idx2txt']))
# #     f.write("---------8----------")
# #     f.write(str(data['txt2idx']))

