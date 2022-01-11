import os

import jieba
import numpy as np


# ========================== 导入数据 ==========================
def read_file(file_path):
    '''
    load the data
    '''
    stopwords = []
    pos = []
    neg = []
    neg_test=[]
    pos_test=[]

    # 读取词库
    inflie = open("语料库/正面评价词语（中文）.txt")
    for line in inflie:
        data = line.rstrip("\n")
        pos.append(data)
    inflie = open("语料库/正面情感词语（中文）.txt")
    for line in inflie:
        data = line.rstrip("\n")
        pos.append(data)

    inflie = open("语料库/负面评价词语（中文）.txt")
    for line in inflie:
        data = line.rstrip("\n")
        neg.append(data)
    inflie = open("语料库/负面情感词语（中文）.txt")
    for line in inflie:
        data = line.rstrip("\n")
        neg.append(data)

    stopwords.append("，")
    with open('stopwords.txt','r',encoding='utf-8') as fr:  
        for i in fr:
            stopwords.append(i.strip())

    # 读入语料库
    for i in range(6000,6001):
        for j in range(0,2999):
            path = file_path+str(i)
            path = os.path.join(path,"neg")
            path = path+"/neg."+str(j)+".txt"
            with open(str(path), 'r',errors="ignore") as f:
                my_data = f.read() # txt中所有字符串读入data，得到的是一个list
                my_data = my_data.rstrip("\n")
                my_data = my_data.replace("\n\n",' ')
                neg_test.append(my_data)
        for j in range(0,2999):
            path = file_path+str(i)
            path = os.path.join(path,"pos")
            path = path+"/pos."+str(j)+".txt"
            with open(str(path), 'r',errors="ignore") as f:
                my_data = f.read() #txt中所有字符串读入data，得到的是一个list
                my_data = my_data.rstrip("\n")
                my_data = my_data.replace("\n\n",' ')
                pos_test.append(my_data)
    eva_neg,eva_pos = sent2word(neg_test,pos_test)
    return neg,pos,eva_neg,eva_pos


# ========================== 文本切割 ==========================
def sent2word(neg_test,pos_test):
    '''
    对每条评论分词,并保存分词结果
    删去停词
    '''
    eva_neg = []
    for i in range(len(neg_test)):
        seg_list = jieba.cut(neg_test[i], cut_all=False)
        seg_list = list(seg_list)
        eva_neg.append(seg_list)

    eva_pos = []
    for i in range(len(pos_test)):
        seg_list = jieba.cut(pos_test[i], cut_all=False)
        seg_list = list(seg_list)
        eva_pos.append(seg_list)

    # for i in range(len(eva_pos)):
    #     for j in eva_pos[i]:
    #         if j in stopwords:
    #             try:
    #                 eva_neg[i].remove(j)
    #             except:
    #                 pass
    # for i in range(len(eva_pos)):
    #     for j in eva_pos[i]:
    #         if j in stopwords:
    #             try:
    #                 eva_pos[i].remove(j)
    #             except:
    #                 pass
    return eva_neg,eva_pos

def GetScore(neg,pos,list):
    '''
    对每一条评论进行打分
    '''
    neg_s = 0
    pos_s = 0
    for w in list:
        if (w in neg) == True:
            neg_s = neg_s + 1
        elif (w in pos) == True:
            pos_s = pos_s + 1
    if (neg_s-pos_s) > 0:
        score = 'NEGATIVE'
        return score
    elif (neg_s-pos_s) < 0:
        score = 'POSITIVE'
        return score

def loss(neg,pos,neg_word,pos_word):
    '''
    计算损失
    '''
    TP,FN,FP,TN=0,0,0,0
    for i in neg:
        if "NEGATIVE" == GetScore(neg_word,pos_word,i):
            TN+=1
        else:
            FP+=1
    for j in pos:
        if 'POSITIVE' == GetScore(neg_word,pos_word,j):
            TP+=1
        else:
            FN+=1

    return TP,FN,FP,TN

def main():
    path ="datasets\ChnSentiCorp_htl_ba_"
    neg_word,pos_word,neg_test,pos_test = read_file(path)
    TP,FN,FP,TN = loss(neg_test,pos_test,neg_word,pos_word)
    acc = (TP+TN)/(len(neg_test)+len(pos_test))
    error = (FN+FP)/(len(neg_test)+len(pos_test))
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    print(TP,FN,FP,TN)
    print("准确率：{:.3f},错误率：{:.3f},精准率：{:.3f},召回率：{:.3f}".format(acc,error,precision,recall))

if __name__=="__main__":
    main()
