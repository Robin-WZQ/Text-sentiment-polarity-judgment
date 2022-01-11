from collections import defaultdict
import os
import re
import jieba
import codecs
 
#生成stopword表，需要去除一些否定词和程度词汇
stopwords = set()
fr = open('停用词.txt','r',encoding='utf-8')
for word in fr:
	stopwords.add(word.strip())#Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
#读取否定词文件
not_word_file = open('否定词.txt','r+',encoding='utf-8')
not_word_list = not_word_file.readlines()
not_word_list = [w.strip() for w in not_word_list]
#读取程度副词文件
degree_file = open('程度副词.txt','r+',encoding='utf-8')
degree_list = degree_file.readlines()
degree_list = [item.split(',')[0] for item in degree_list]
#生成新的停用词表
with open('stopwords.txt','w',encoding='utf-8') as f:
	for word in stopwords:
		if(word not in not_word_list) and (word not in degree_list):
			f.write(word+'\n')
 
def seg_word(sentence):
	'''
	jieba分词后去除停用词
	'''
	seg_list = jieba.cut(sentence)
	seg_result = []
	for i in seg_list:
		seg_result.append(i)
	stopwords = set()
	with open('停用词.txt','r',encoding='utf-8') as fr:
		for i in fr:
			stopwords.add(i.strip())
	return list(filter(lambda x :x not in stopwords,seg_result))		
 

def classify_words(word_list):
	'''
	找出文本中的情感词、否定词和程度副词
	'''
	sen_file = open('BosonNLP_sentiment_score.txt','r+',encoding='utf-8')
	sen_list = sen_file.readlines()
	sen_dict = defaultdict()
	for i in sen_list:
		if len(i.split(' '))==2:
			sen_dict[i.split(' ')[0]] = i.split(' ')[1]

	not_word_file = open('否定词.txt','r+',encoding='utf-8')
	not_word_list = not_word_file.readlines()
	degree_file = open('程度副词.txt','r+',encoding='utf-8')
	degree_list = degree_file.readlines()

	degree_dict = defaultdict()
	for i in degree_list:
		if len(i.split(' '))==2:        
			degree_dict[i.split(',')[0]] = i.split(',')[1]
 
	sen_word = dict()
	not_word = dict()
	degree_word = dict()

	for i in range(len(word_list)):
		word = word_list[i]
		if word in sen_dict.keys() and word not in not_word_list and word not in degree_dict.keys():
			sen_word[i] = sen_dict[word]
		elif word in not_word_list and word not in degree_dict.keys():
			not_word[i] = -1
		elif word in degree_dict.keys():
			degree_word[i]  = degree_dict[word]
 
	sen_file.close()
	not_word_file.close()
	degree_file.close()

	return sen_word,not_word,degree_word
 
def score_sentiment(sen_word,not_word,degree_word,seg_result):
	'''
	计算情感词的分数
	'''
	W = 1
	score = 0
	sentiment_index = -1
	sentiment_index_list = list(sen_word.keys())
	for i in range(0,len(seg_result)):
		if i in sen_word.keys():
			score += W*float(sen_word[i])
			sentiment_index += 1
			if sentiment_index < len(sentiment_index_list)-1:
				for j in range(sentiment_index_list[sentiment_index],sentiment_index_list[sentiment_index+1]):
					if j in not_word.keys():
						W *= -1
					elif j in degree_word.keys():
						W *= float(degree_word[j])	
		if sentiment_index < len(sentiment_index_list)-1:
			i = sentiment_index_list[sentiment_index+1]
	return score
 
 
#计算得分
def sentiment_score(sentence):
	#1.对文档分词
	seg_list = seg_word(sentence)
	#2.将分词结果转换成字典，找出情感词、否定词和程度副词
	sen_word,not_word,degree_word = classify_words(seg_list)
	#3.计算得分
	score = score_sentiment(sen_word,not_word,degree_word,seg_list)
	return score

# ========================== 导入数据 ==========================
def read_file(file_path):
    '''
    load the data
    '''
    neg_test=[]
    pos_test=[]

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
    return neg_test,pos_test

def loss(neg,pos):
    TP,FN,FP,TN=0,0,0,0
    for i in neg:
        if sentiment_score(i)<0:
            TN+=1
        else:
            FP+=1
    for j in pos:
        if sentiment_score(j)>0:
            TP+=1
        else:
            FN+=1

    return TP,FN,FP,TN

def main():
    path ="datasets\ChnSentiCorp_htl_ba_"
    neg_test,pos_test = read_file(path)
    TP,FN,FP,TN = loss(neg_test,pos_test)
    acc = (TP+TN)/(len(neg_test)+len(pos_test))
    error = (FN+FP)/(len(neg_test)+len(pos_test))
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    print(TP,FN,FP,TN)
    print("准确率：{:.3f},错误率：{:.3f},精准率：{:.3f},召回率：{:.3f}".format(acc,error,precision,recall))

    # print("标准间太差 而且设施非常陈旧    ",sentiment_score("标准间太差"))
    # print('入住时被告知是两间豪标双人间，其实想要一大一双。总之不尽如人意。   ',sentiment_score('总之不尽如人意。'))
    # print('商务大床房，房间很大，床有2M宽，整体感觉经济实惠不错!    ',sentiment_score('商务大床房，房间很大，床有2M宽，整体感觉经济实惠不错!'))
    # print('房间内环境还是不错的,就是上网有点贵    ',sentiment_score('房间内环境还是不错的,就是上网有点贵,'))

if __name__=="__main__":
    main()
 