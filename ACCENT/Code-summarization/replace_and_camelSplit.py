
import re
def replace_a_to_b(code,var_a,var_b):
    str_list=code.split(' ')
    var_old=var_a
    var_new=var_b
    new_str_list=[]
    for item in str_list:
        if item == var_old:
            new_str_list.append(var_new)
        else:
            new_str_list.append(item)
    '''
    code_new=new_str_list[0]
    for i in range(len(new_str_list)-1):
        code_new=code_new+' '+new_str_list[i+1]
    return code_new
    '''
    return new_str_list

#将token切分成subtoken
def hasZM(token):
    regex = "[a-zA-Z]"
    pattern = re.compile(regex)
    result = pattern.findall(token)
    if len(result) == 0:
        return False

def camelSplit(token):
    result=hasZM(token)
    if result==False:   ###没有字母 直接返回token  例如['+']
        return [token]
    if "_" in token:
        sub=[]
        subTokens = token.split("_")
        for item in subTokens:
            result_=hasZM(item)
            if result_==False:
                sub=sub+[item]
                continue
            if item.isupper():
                sub=sub+[item]
            else:
                newTok=item[0].upper()+item[1:]
                regex = "[A-Z][a-z]*\d*[a-z]*"
                pattern = re.compile(regex)
                sub_temp = pattern.findall(newTok)
                if sub_temp==[]:
                    sub=sub+[item]
                    continue
                sub_temp[0]=item[0]+sub_temp[0][1:]
                sub=sub+sub_temp
        return sub

    elif token.isupper():
        return [token]
    else:
        newToken = token[0].upper() + token[1:]
        regex = "[A-Z][a-z]*\d*[a-z]*"
        pattern = re.compile(regex)
        subTokens = pattern.findall(newToken)
        if subTokens==[]:
            return [newToken]
        subTokens[0] = token[0] + subTokens[0][1:]
    return subTokens

def split_c_and_s(code_list):
    new_code_list=[]
    for item in code_list:
        sub=camelSplit(item)
        for char in sub:
            new_code_list.append(char)

    code_new = new_code_list[0]

    print(new_code_list[-1])
    if '\n' not in new_code_list[-1]:
        new_code_list.append('\n')
        print('append')

    for i in range(len(new_code_list) - 1):
        code_new = code_new + ' ' + new_code_list[i + 1]
    return code_new
import time










