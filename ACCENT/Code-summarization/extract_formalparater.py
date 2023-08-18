import javalang as javalang
import pandas as pd
import os
import time
from utils import get_sequence as func

def trans_to_sequences(ast):
    sequence = []
    func(ast, sequence)
    return sequence

def extract_var_name(file,root,save_path):
    f_code = open(file, 'r')

    var_list = []
    index_list = []
    count = 0
    err = 0
    errlist = []
    for line in f_code:

        code = line
        formalpara = []
        tokens = javalang.tokenizer.tokenize(code)

        # tree = parser.parse_member_declaration()
        try:
            parser = javalang.parser.Parser(tokens)
            tree = parser.parse_member_declaration()
        except:
            err+=1
            count += 1
            errlist.append(count)

            print("pass")
            continue
        ast_list = trans_to_sequences(tree)

        for i in range(len(ast_list)):
            item = ast_list[i]
            if item == 'FormalParameter':
                formalpara.append(ast_list[i + 3])

        print(formalpara)
        var_list.append(formalpara)
        index_list.append(count)
        count = count + 1
        print('ok  ' + str(count))
    print("err",err)
    print("errlist",errlist)


    data_var = pd.DataFrame({'id': index_list, 'variable': var_list})
    data_var.to_pickle(save_path)


if __name__=='__main__':
    root = '../Code-summarization'
    totaltime = 0
    totaltime2 = 0
    start_num = 11
    end_num = 12
    for mycnt in range(start_num, end_num):
        start = time.process_time()
        start2 = time.time()
        print('start at time:')
        print(time.strftime(' %Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        file=root+'/data1/java' + str(mycnt) + '/test1/code.original'
        save_path=root+'/data1/java' + str(mycnt) + '/formalParameter_for_everyCode_test1.pkl'
        print('extract var name :')

        extract_var_name(file,root,save_path)
        print('extract end!')
        print('end...')
        print(time.strftime(' %Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        # end = time.process_time()
        end = time.process_time()
        end2 = time.time()
        totaltime += end - start
        totaltime2 += end2 - start2
        print("time cost2: ", end2 - start2)
