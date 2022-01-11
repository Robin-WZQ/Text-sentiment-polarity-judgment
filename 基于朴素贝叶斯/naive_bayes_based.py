from collections import defaultdict
import os
import jieba
import time
import math

negtive_word_hash = {}
positive_word_hash = {}

def new_stopwords():
    '''
    生成stopword表，需要去除一些否定词和程度词汇（只需运行一次）
    '''
    stopwords = set()
    fr = open('停用词.txt','r',encoding='utf-8')
    for word in fr:
        stopwords.add(word.strip())#Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
    #读取否定词文件
    not_word_file = open('否定词.txt','r+',encoding='utf-8')
    not_word_list = not_word_file.readlines()
    not_word_list = [w.strip() for w in not_word_list]
    #生成新的停用词表
    with open('stopwords.txt','w',encoding='utf-8') as f:
        for word in stopwords:
            if(word not in not_word_list):
                f.write(word+'\n')

def read_file(file_path):
    '''
    load the data
    '''
    neg_test=[]
    pos_test=[]

    # 读入语料库
    for i in range(6000,6001):
        for j in range(0,500):
            path = file_path+str(i)
            path = os.path.join(path,"neg")
            path = path+"/neg."+str(j)+".txt"
            with open(str(path), 'r',errors="ignore") as f:
                my_data = f.read() # txt中所有字符串读入data，得到的是一个list
                my_data = my_data.rstrip("\n")
                my_data = my_data.replace("\n\n",' ')
                neg_test.append(my_data)
        for j in range(0,500):
            path = file_path+str(i)
            path = os.path.join(path,"pos")
            path = path+"/pos."+str(j)+".txt"
            with open(str(path), 'r',errors="ignore") as f:
                my_data = f.read() #txt中所有字符串读入data，得到的是一个list
                my_data = my_data.rstrip("\n")
                my_data = my_data.replace("\n\n",' ')
                pos_test.append(my_data)
    return neg_test,pos_test

def seg_word(sentence):
    '''
    jieba分词后去除停用词
    '''
    seg_list = jieba.cut(sentence)
    seg_result = []
    for i in seg_list:
        seg_result.append(i)
    stopwords = set()
    with open('stopwords.txt','r',encoding='utf-8') as fr:
        for i in fr:
            stopwords.add(i.strip())
    return list(filter(lambda x :x not in stopwords,seg_result))		

 
def classifier(neg,pos):
    '''
    naive bayes classifier
    :param seg_list -> words of sentence(processing bu jieba)
    '''
    # =================== 获取所有词 ===================
    negtive_words = []
    positive_words = []
    negtive_sentence = []
    positive_sentence = []
    for i in neg:
        words = seg_word(i)
        negtive_sentence.append(words)
        for j in words:
            negtive_words.append(j)
    for i in pos:
        words = seg_word(i)
        positive_sentence.append(words)
        for j in words:
            positive_words.append(j)

    # # =================== 计算出现频率 ===================
    # neg_len = len(negtive_words)
    # pos_len = len(positive_words)
    # for i in negtive_words: 
    #     if negtive_word_hash.get(i) == None:
    #         num = negtive_words.count(i)
    #         frq = num/neg_len
    #         negtive_word_hash[i] = frq
    # for i in positive_words:
    #     if positive_word_hash.get(i) == None:
    #         num = positive_words.count(i)
    #         frq = num/pos_len
    #         positive_word_hash[i] = frq

   # =================== 特征计算 ===================
    neg_len = len(neg)
    pos_len = len(pos)
    N = neg_len+pos_len
    for i in negtive_words:
        A,B,C,D =0,0,0,0
        if i not in negtive_word_hash:
            for j in range(len(negtive_sentence)):
                if negtive_sentence[j].count(i)!=0:
                    A= A + 1
            for j in positive_sentence:
                if i in j:
                    B+=1
            C = neg_len - A
            D = pos_len - B
            M = math.pow((A*D-B*C),2)
            CHI = (N*M)/((A+C)*(B+D)*(A+B)*(C+D))
            if CHI == 0:
                negtive_word_hash[i] = 1e-8
            else:
                negtive_word_hash[i] = CHI * 0.01

    for i in positive_words:
        A,B,C,D =0,0,0,0
        if i not in positive_word_hash:
            for j in positive_sentence:
                if i in j:
                    A+=1
            for j in negtive_sentence:
                if i in j:
                    B+=1
            C = pos_len - A
            D = neg_len - B
            M = math.pow((A*D-B*C),2)
            CHI = (N*M)/((A+C)*(B+D)*(A+B)*(C+D))
            if CHI == 0:
                 positive_word_hash[i] = 1e-8
            else:
                positive_word_hash[i] = CHI * 0.01


def score_sentece(sentence):
    '''
    compute the 极性 of a sentence
    '''
    sen_word = seg_word(sentence)
    pos_score = 0
    neg_score = 0
    for word in sen_word:
        # if (word in positive_word_hash) and word!=' ':
        #     pos_score += math.log(positive_word_hash[word])
        # else:
        #     pos_score += 0

        # if (word in negtive_word_hash)  and word!=' ':
        #     neg_score += math.log(negtive_word_hash[word])
        # else:
        #     neg_score+=0
        try:
            pos_score += math.log(positive_word_hash[word])
            neg_score += math.log(negtive_word_hash[word])
        except:
            pass
    return pos_score - neg_score

def loss(neg,pos):
    TP,FN,FP,TN=0,0,0,0
    for j in pos:
        if score_sentece(j)>0:
            TP+=1
        else:
            FN+=1
    for i in neg:
        if score_sentece(i)<0:
            TN+=1
        else:
            FP+=1
    return TP,FN,FP,TN


def main():
    #new_stopwords()
    path ="datasets\ChnSentiCorp_htl_ba_"
    neg_test,pos_test = read_file(path)
    print("读取数据完成")
    begin_time = time.time()
    classifier(neg_test,pos_test)
    end_time = time.time()
    print("训练用时{}s".format(round(end_time-begin_time,2)))
    TP,FN,FP,TN = loss(neg_test,pos_test)
    acc = (TP+TN)/(len(neg_test)+len(pos_test))
    error = (FN+FP)/(len(neg_test)+len(pos_test))
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    print(TP,FN,FP,TN)
    print("准确率：{:.3f},错误率：{:.3f},精准率：{:.3f},召回率：{:.3f}".format(acc,error,precision,recall))

if __name__=="__main__":
    main()
 