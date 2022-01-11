from logging import logProcesses
from gensim.models.word2vec import Word2Vec
import os
import jieba
import random
from train import train
from model.logistic import logistic_net
from torchvision import datasets
from torch.nn import init
import torch.nn
import torch
import torch.optim
import torch.utils

def read_file(file_path):
    '''
    load the data
    '''
    all = []

    # 读入语料库
    i = 6000
    for j in range(0,3000):
        path = file_path+str(i)
        path = os.path.join(path,"neg")
        path = path+"/neg."+str(j)+".txt"
        with open(str(path), 'r',errors="ignore") as f:
            my_data = f.read() # txt中所有字符串读入data，得到的是一个list
            my_data = my_data.rstrip("\n")
            my_data = my_data.replace("\n\n",' ')
            my_data = (seg_word(my_data),0)
            all.append(my_data)
    for j in range(0,3000):
        path = file_path+str(i)
        path = os.path.join(path,"pos")
        path = path+"/pos."+str(j)+".txt"
        with open(str(path), 'r',errors="ignore") as f:
            my_data = f.read() #txt中所有字符串读入data，得到的是一个list
            my_data = my_data.rstrip("\n")
            my_data = my_data.replace("\n\n",' ')
            my_data = (seg_word(my_data),1)
            all.append(my_data)

    return all

def data_split(data,model):
    '''
    split the data for training, validating and testing
    '''
    new = []
    for i,x in enumerate(data):
        sentence = x[0]
        label = x[1]
        init = torch.zeros(size=[300])
        for word in sentence:
            try:
                init += model.wv[word]
            except:
                pass
        new.append((init,label))
    lenth_data = len(new)
    random.shuffle(data)
    train_dataset = new[:int(0.9*lenth_data)]
    val_dataset = new[int(0.8*lenth_data):int(0.9*lenth_data)]
    test_dataset = new[int(0.9*lenth_data):]

    return train_dataset,val_dataset,test_dataset

def seg_word(sentence):
    '''
    jieba分词后去除停用词
    '''
    seg_list = jieba.cut(sentence)
    seg_result = []
    for i in seg_list:
        seg_result.append(i)
    stopwords = set()
    with open('data\stopwords.txt','r',encoding='utf-8') as fr:
        for i in fr:
            stopwords.add(i.strip())
    return list(filter(lambda x :x not in stopwords,seg_result))

def main():
    word_model = Word2Vec.load("my_word2vec")
    path ="datasets\ChnSentiCorp_htl_ba_"
    device = "cuda"
    print("开始读取数据")
    all = read_file(path)
    print("读取数据完毕")
    train_dataset,val_dataset,test_dataset = data_split(all,word_model)
    net = logistic_net()

    for name, param in net.named_parameters():
        if 'weight' in name:
            init.normal_(param, mean=0, std=0.01)
        if 'bias' in name:
            init.constant_(param, val=1)

    optimizor= torch.optim.SGD(net.parameters(),lr=0.04,momentum=0.9)
    criterion = torch.nn.BCELoss(size_average=False)

    n_epoches = 500
    train_batch_size = 100
    test_batch_size = 1
    #建立一个数据迭代器
    # 装载训练集
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=train_batch_size,
                                            shuffle=True)
    # 装载验证集
    test_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=test_batch_size,
                                            shuffle=True)
    # 装载测试集
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=test_batch_size,
                                            shuffle=True)

    train('cpu',train_loader,optimizor,criterion,n_epoches,train_batch_size,net,test_loader)

    print("Done!")

if __name__ == "__main__":
    main()