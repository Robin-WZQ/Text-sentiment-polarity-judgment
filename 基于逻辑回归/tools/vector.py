import os
import pandas as pd
# from nlpia.loaders import get_data
# from gensim.models.word2vec import KeyedVectors

import jieba
from gensim.models.word2vec import Word2Vec


def new_stopwords():
    '''
    生成stopword表，需要去除一些否定词和程度词汇（只需运行一次）
    '''
    stopwords = set()
    fr = open('data/停用词.txt','r',encoding='utf-8')
    for word in fr:
        stopwords.add(word.strip())#Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
    #读取否定词文件
    not_word_file = open('data/否定词.txt','r+',encoding='utf-8')
    not_word_list = not_word_file.readlines()
    not_word_list = [w.strip() for w in not_word_list]
    #生成新的停用词表
    with open('data/stopwords.txt','w',encoding='utf-8') as f:
        for word in stopwords:
            if(word not in not_word_list):
                f.write(word+'\n')

def read_file(file_path):
    '''
    load the data
    '''
    neg = []
    pos = []

    # 读入语料库
    i = 6000
    for j in range(0,2999):
        path = file_path+str(i)
        path = os.path.join(path,"neg")
        path = path+"/neg."+str(j)+".txt"
        with open(str(path), 'r',errors="ignore") as f:
            my_data = f.read() # txt中所有字符串读入data，得到的是一个list
            my_data = my_data.rstrip("\n")
            my_data = my_data.replace("\n\n",' ')
            neg.append(my_data)
    for j in range(0,2999):
        path = file_path+str(i)
        path = os.path.join(path,"pos")
        path = path+"/pos."+str(j)+".txt"
        with open(str(path), 'r',errors="ignore") as f:
            my_data = f.read() #txt中所有字符串读入data，得到的是一个list
            my_data = my_data.rstrip("\n")
            my_data = my_data.replace("\n\n",' ')
            pos.append(my_data)

    return neg,pos


def seg_word(sentence):
    '''
    jieba分词后去除停用词
    '''
    seg_list = jieba.cut(sentence)
    seg_result = []
    for i in seg_list:
        seg_result.append(i)
    stopwords = set()
    with open('data/stopwords.txt','r',encoding='utf-8') as fr:
        for i in fr:
            stopwords.add(i.strip())
    return list(filter(lambda x :x not in stopwords,seg_result))		


def got_word(neg,pos):
    # =================== 获取所有句子 ===================
    word = []
    for i in neg:
        words = seg_word(i)
        word.append(words)
    for i in pos:
        words = seg_word(i)
        word.append(words)
    
    return word

def model_train(token_list):
    num_features = 300
    min_word_count = 3
    num_workers = 1
    window_size = 3
    subsampling = 1e-3

    model = Word2Vec(
        token_list,
        workers=num_workers,
        vector_size=num_features,
        min_count=min_word_count,
        window=window_size,
        sample=subsampling,
        epochs=100,
        sg=1
    )

    model.init_sims(replace=True)
    model_name = "my_word2vec_skip"
    model.save(model_name)

    return True
    
def main():
    # #new_stopwords() #只需运行一次
    path ="E:\大三\自然语言处理\作业\code\datasets\ChnSentiCorp_htl_ba_"
    neg,pos = read_file(path)
    token_list = got_word(neg,pos)
    if(model_train(token_list)):
        print("训练完成")

    model = Word2Vec.load("my_word2vec_skip")

    for e in model.wv.most_similar(positive=['脏'], topn=10):
        print(e[0], e[1])

if __name__=="__main__":
    main()
 