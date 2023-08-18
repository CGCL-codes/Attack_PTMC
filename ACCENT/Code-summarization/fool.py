import pandas as pd
import time
SUBSTITUTION=3
def generate_adv(best_sub_path,f_code_origin_path,f_code_new_path):
    best_sub=pd.read_pickle(best_sub_path)
    print(best_sub)
    print("here\n")
    best_sub_list=best_sub['var_sub'].tolist()

    f_code_origin=open(f_code_origin_path,'r')
    f_code_new=open(f_code_new_path,'w')

    count=0
    for line in f_code_origin:
        best_sub_dict=best_sub_list[count]
        old_list=[]
        new_list=[]
        for k,v in best_sub_dict.items():
            old_list.append(k)
            new_list.append(v)
        # if len(old_list)>SUBSTITUTION:
        #     old_list=old_list[0:SUBSTITUTION]
        #     new_list=new_list[0:SUBSTITUTION]
        code=line.split()
        new_code=[]
        for token in code:
            if token in old_list:
                index=old_list.index(token)
                new_code.append(new_list[index])  #xin de bian liang ming
            else:
                new_code.append(token)

        line_new = new_code[0]
        for i in range(len(new_code) - 1):
            line_new = line_new + ' ' + new_code[i + 1]
        f_code_new.write(line_new+'\n')
        count=count+1

from replace_and_camelSplit import split_c_and_s
def generate_adv_subtoken(f_code_new_path,f_code_sub_new_path):
    f=open(f_code_new_path,'r')
    f_new=open(f_code_sub_new_path,'w')

    for line in f:

        code=line.split(' ')
        line_new=split_c_and_s(code)
        f_new.write(line_new)
        print('ok')


root = '../Code-summarization'
# name = "/CodeBert"
name = "/PLBART"
# name = "/CodeGPT"


totaltime = 0
totaltime2 = 0
start_num = 11
end_num = 12
for mycnt in range(start_num, end_num):
    start = time.process_time()
    start2 = time.time()
    print(time.strftime(' %Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    f_code_origin_path = root + '/data1/java' + str(mycnt) + '/test/code.original'
    root += name
    best_sub_path=root+'/data/java' + str(mycnt) + '/test_rnn_data_descend_best_var.pkl'
    f_code_new_path=root+'/data/java' + str(mycnt) + '/test/code_adv3_rnn.original'
    f_code_sub_new_path=root+'/data/java' + str(mycnt) + '/test/code_adv3_rnn.original_subtoken'
    generate_adv(best_sub_path,f_code_origin_path,f_code_new_path)
    generate_adv_subtoken(f_code_new_path,f_code_sub_new_path)
    print('end...')
    print(time.strftime(' %Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    end = time.process_time()
    end2 = time.time()
    totaltime += end - start
    totaltime2 += end2 - start2
    print("time cost2: ", end2 - start2)



