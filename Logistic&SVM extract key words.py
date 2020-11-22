
# coding: utf-8

# In[151]:


import csv
import jieba
import re

pattern1 =   '(胸廓).{,}(改变|增大|破坏|塌陷|小|对称|不对称|骨折|不均匀|均匀|多发|畸形)' 
pattern2 =   '(欠均匀|未见明显|炎|不均|欠均|高|低|增|多发|无明显|明显|随诊)(密度).{,}(未见明显|炎|不均|欠均|高|低|增|多发|无明显|明显|随诊)' 
pattern3 =   '(右肺).{,}(炎|实变|结节|气肿|索条|病变|增|减|复查|通畅|恶性|转移|缓解|积液|骨折|未见明显)' 
pattern4 =   '(气管).{,}(炎|痰|气囊肿|感染|改变|扩张|实变|不对称|骨折|阻塞|肿大|高密度|异物|结节|增|多发|闭塞|淋巴结|移|隆起|减|不通畅)' 
pattern5 =   '(纹理).{,}(增|炎|欠均匀|紊乱|复查|重)' 
pattern6 =   '(肺野).{,}(不均|欠均|增|减|略|炎|未见|结节)' 
pattern7 =   '(血管).{,}(瘤|炎|畸形|囊肿|低密度|积气|结石|提高|不均|增|扩张|硬化|未见异常|淋巴结|大|小|病变|积液|血肿|迂曲|不清|移位|断面 )'
pattern8 =   '(液性).{,}(炎|积液|减|增)' 
def select_words(pattern):
    list1=[]
    list2=[]
    dict={}
    with open('e:/train.csv','rt') as csvfile:
        reader = csv.reader(csvfile)
        rows=[ row[17] for row in reader]
        for i in range(0,len(rows)):
            list1.append( ''.join(re.findall(u'[\u4e00-\u9fff]+', rows[i])))
        for i in range(0,len(list1)):
            content = list1[i]#清洗后的文档
            find = re.findall(pattern,content)
            for word in find:
                if word not in dict:
                    dict[word]=1
                else:
                    dict[word]+=1                                                                       #放入字典
        list2  = sorted(dict.items(),key=lambda x:x[1],reverse=True) 
        print (list2)
        print ("\n")
if __name__ == '__main__':
    select_words(pattern1)
    select_words(pattern2)
    select_words(pattern3)
    select_words(pattern4)
    select_words(pattern5)
    select_words(pattern6)
    select_words(pattern7)
    select_words(pattern8)

