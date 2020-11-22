# coding=utf-8
# -*- coding: cp936 -*-
import re
import sys
import jieba
import jieba.posseg as pseg
import jieba.analyse
print sys.getdefaultencoding()
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#分别计算阴性、阳性的tf-idf 'd:yin.xls' 'd:yang.xls'
def calculate_tfidf(d):
    word_str0=[]
    word_str0=d[u'描述']
    word_str = ' '.join(word_str0)
    word_str = re.sub(r'\t+', ' ', word_str)  # trans Tab to空格
    word_str = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——；！，”。《》，。：“？、~@#￥%……&*（）1234567①②③④)]+".\
                          decode("utf8"), "".decode("utf8"), word_str)
    #print word_str1
    result =  jieba.analyse.extract_tags(word_str,50)
    #for t in result:
        #print t 
    return result
#分别计算阴性、阳性的名词tf-idf  calculate_tfidf('d:yin.xls' )  calculate_tfidf('d:yang.xls' )
def calculate_tfidfN(d):
    word_str0=[]
    word_str0=d[u'描述']
    word_str = ' '.join(word_str0)
    word_str = re.sub(r'\t+', ' ', word_str)  # trans Tab to空格
    word_str = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——；！，”。《》，。：“？、~@#￥%……&*（）1234567①②③④)]+".\
                          decode("utf8"), "".decode("utf8"), word_str)
    words = pseg.cut(word_str)
    word_list = []
    for wds in words:
        if wds.flag == 'x' and wds.word != ' ' and wds.word != 'ns' \
                        or re.match(r'^n', wds.flag) != None \
                                and re.match(r'^nr', wds.flag) == None:
                    word_list.append(wds.word)
    word_str1 = ''.join(word_list)
    #print word_str3
    result =  jieba.analyse.extract_tags(word_str1,50)
    #for t in result:
        #print t  
    return result
#清洗文档
def deal_text(path):
   data = pd.read_excel(path)
   d = pd.DataFrame(data)
   return d
#处理性别,男1 女0
def deal_sex(d0):
    aa = []
    sex = np.zeros(d0.shape[0])
    for indexs in d0.index:
        aa = d0.loc[indexs]
        if re.search(u'男',aa[u'性别']) is None:
            sex[indexs] = 0
        else:
            sex[indexs] = 1
    return sex
#处理年龄,月、天均转化为年
def deal_age(d0):
    cc = []
    age = np.zeros(d0.shape[0])
    for indexs in d0.index:
        cc = d0.loc[indexs]
        if(re.search(u'岁',cc[u'年龄']) is not None):
            age[indexs] = float(cc[u'年龄'][0:-1])
        if(re.search(u'月',cc[u'年龄']) is not None):
            age[indexs] = float(cc[u'年龄'][0:-1])/12
        if(re.search(u'天',cc[u'年龄']) is not None):
            age[indexs] = float(cc[u'年龄'][0:-1])/365
    return age
#处理胸壁
def Xiongbi_analyse(d0):
    Xiongbi = np.zeros(d0.shape[0])
    for indexs in d0.index:
        aa = d0.loc[indexs]
        if re.search(u'胸壁',aa[u'描述']) is None:
            Xiongbi[indexs] = 0
        else:
            if re.search(u'胸壁光滑',aa[u'描述']) is None:
                Xiongbi[indexs] = 1
            else:
                Xiongbi[indexs] = 0
    print'胸壁数据记录完毕'
    return Xiongbi
#处理骨头
def Gutou_analyse(d0):
    Gutou = np.zeros(d0.shape[0])
    for indexs in d0.index:
        aa = d0.loc[indexs]
        if re.search(u'骨.{,6}[断折]',aa[u'描述']) is None:
            Gutou[indexs] = 0
        else:
            if re.search(u'未见.{,4}[断折]',aa[u'描述']) is None:
                Gutou[indexs] = 1
            else:
                Gutou[indexs] = 0
    print'骨头数据记录完毕'
    return Gutou
#处理支气管
def Zhiqiguan_analyse(d0):
    Zhiqiguan = np.zeros(d0.shape[0])
    for indexs in d0.index:
        aa = d0.loc[indexs]
        if re.search(u'支气管',aa[u'描述']) is None:
            Zhiqiguan[indexs] = 0
        else:
            if re.search(u'支气管.{,2}通畅',aa[u'描述']) is None:
                Zhiqiguan[indexs] = 1
            else:
                if re.search(u'支气管.{,2}扩张',aa[u'描述']) is None:
                    Zhiqiguan[indexs] = 0
                else:
                    Zhiqiguan[indexs] = 1
    print'气管数据记录完毕'
    return Zhiqiguan
#%%
def Wenli_analyse(d0):
    Wenli = np.zeros(d0.shape[0])
    for indexs in d0.index:
        aa = d0.loc[indexs]
        if re.search(u'纹理',aa[u'描述']) is None:
            Wenli[indexs] = 0
        else:
            if re.search(u'纹理.{,2}清晰',aa[u'描述']) is None:
                Wenli[indexs] = 1
            else:
                Wenli[indexs] = 0
    print'纹理数据记录完毕'
    return Wenli
#处理液性病变
def Ye_analyse(d0):
    count2 = 0
    Ye = np.zeros(d0.shape[0])
    for indexs in d0.index:
        aa = d0.loc[indexs]
        if re.search(u'液.[体性]',aa[u'描述']) is None:
            Ye[indexs] = 0
        else:
            Ye[indexs] = 1
        if re.search(u'积液',aa[u'描述']) is None:
            count2+=1
        else:
            if re.search(u'积液.{,2}吸收',aa[u'描述']) is None:
                Ye[indexs] = 1
    print'液性数据记录完毕'
    return Ye   
#处理肺门数据
def Feimen_analyse(d0):
    Feimen = np.zeros(d0.shape[0])
    for indexs in d0.index:
        aa = d0.loc[indexs]
        if re.search(u'肺门影',aa[u'描述']) is None:
            Feimen[indexs] = 0
        else:
            if re.search(u'肺门影.{,2}不大',aa[u'描述']) is None:
                Feimen[indexs] = 1
            else:
                if re.search(u'肺门影.{,2}[增浓]',aa[u'描述']) is None:
                    Feimen[indexs] = 0    
                else:
                    Feimen[indexs] = 1
    print'肺门数据记录完毕'
    return Feimen
#处理病变影
def Ying_analyse(d0):
    Ying = np.zeros(d0.shape[0])
    for indexs in d0.index:
        aa = d0.loc[indexs]
        if re.search(u'[斑片条索].影',aa[u'描述']) is None:
            Ying[indexs] = 0
        else:
            if re.search(u'[斑片].影',aa[u'描述']) is None:
                Ying[indexs] = 1
            else:
                Ying[indexs] = 2
    print'影数据记录完毕'
    return Ying
#处理结节数据
def Jiejie_analyse(d0):
    Jiejie = np.zeros(d0.shape[0])
    for indexs in d0.index:
        aa = d0.loc[indexs]
        if re.search(u'[肺间隔].{,8}结节',aa[u'描述']) is None:
            Jiejie[indexs] = 0
        else:
            if re.search(u'结节.消失',aa[u'描述']) is None:
                Jiejie[indexs] = 1
            else:
                Jiejie[indexs] = 0
    print'结节数据记录完毕'
    return Jiejie
#处理透光数据
def Guang_analyse(d0):
    Guang = np.zeros(d0.shape[0])
    for indexs in d0.index:
        aa = d0.loc[indexs]
        if re.search(u'透.[光亮].区',aa[u'描述']) is None:
            Guang[indexs] = 0
        else:
            Guang[indexs] = 1
    print'透光数据记录完毕'
    return Guang
#处理钙化数据
def Gai_analyse(d0):
    Gai = np.zeros(d0.shape[0])
    for indexs in d0.index:
        aa = d0.loc[indexs]
        if re.search(u'[肺间隔].{,8}钙化',aa[u'描述']) is None:
            Gai[indexs] = 0
        else:
            Gai[indexs] = 1
    print'钙化数据记录完毕'
    return Gai
#记录病变实情
def yin_yang(d0): 
    Bing = np.zeros(d0.shape[0])
    for indexs in d0.index:
        aa = d0.loc[indexs]
        if re.search(u'阴',aa[u'阳性']) is None:
            Bing[indexs] = 0
        else:
            Bing[indexs] = 1
    return Bing
 # coding=utf-8
# -*- coding: cp936 -*-
def get_data(d0):
    train_data = pd.DataFrame({'age':deal_age(d0),'sex':deal_sex(d0),'Xiongbi':Xiongbi_analyse(d0),'Gutou':Gutou_analyse(d0),'Zhiqiguan':Zhiqiguan_analyse(d0),'Wenli':Wenli_analyse(d0),'Ye':Ye_analyse(d0),'Feimen':Feimen_analyse(d0),'Ying':Ying_analyse(d0),'Jiejie':Jiejie_analyse(d0),'Guang':Guang_analyse(d0),'Gai':Gai_analyse(d0),'Bing':yin_yang(d0)})
    count = 0
    for indexs in train_data.index:
        aa = d0.loc[indexs]
        if re.search(u'未见异常',aa[u'诊断']) is None:
            count+=1
        else:
            train_data = train_data.drop(indexs)
    print(count)
    return train_data
def get_data2(d0):
    train_data = pd.DataFrame({'age':deal_age(d0),'sex':deal_sex(d0),'Xiongbi':Xiongbi_analyse(d0),'Gutou':Gutou_analyse(d0),'Zhiqiguan':Zhiqiguan_analyse(d0),'Wenli':Wenli_analyse(d0),'Ye':Ye_analyse(d0),'Feimen':Feimen_analyse(d0),'Ying':Ying_analyse(d0),'Jiejie':Jiejie_analyse(d0),'Guang':Guang_analyse(d0),'Gai':Gai_analyse(d0),'Bing':yin_yang(d0)})
    return train_data
def get_data_test(d0):
    train_data = pd.DataFrame({'age':deal_age(d0),'sex':deal_sex(d0),'Xiongbi':Xiongbi_analyse(d0),'Gutou':Gutou_analyse(d0),'Zhiqiguan':Zhiqiguan_analyse(d0),'Wenli':Wenli_analyse(d0),'Ye':Ye_analyse(d0),'Feimen':Feimen_analyse(d0),'Ying':Ying_analyse(d0),'Jiejie':Jiejie_analyse(d0),'Guang':Guang_analyse(d0),'Gai':Gai_analyse(d0)})
    return train_data
def calculate_accuracy(train_data):
    predictors = ["age","sex", "Xiongbi", "Gutou", "Zhiqiguan", "Wenli", "Ye","Feimen","Ying","Jiejie","Guang","Gai"]
    results = []
    sample_leaf_options = list(range(1, 500, 10))
    n_estimators_options = list(range(1, 1000, 50))
    groud_truth = train_data['Bing'][15001:]

    for leaf_size in sample_leaf_options:
        for n_estimators_size in n_estimators_options:
            alg = RandomForestClassifier(min_samples_leaf=leaf_size, n_estimators=n_estimators_size, random_state=50)
            alg.fit(train_data[predictors], train_data['Bing'])
            predict = alg.predict(train_data[predictors][15001:])
            # 用一个三元组，分别记录当前的 min_samples_leaf，n_estimators， 和在测试数据集上的精度
            results.append((leaf_size, n_estimators_size, (groud_truth == predict).mean()))
            print((leaf_size, n_estimators_size, (groud_truth == predict).mean()))       # 真实结果和预测结果进行比较，计算准确率
    # 打印精度最大的那一个三元组
    print(max(results, key=lambda x: x[2]))
    print results 
if __name__ == '__main__':
    d0 = deal_text('d:zong.xls')
    d1 = deal_text('d:yin.xls')
    d2 = deal_text('d:yang.xls')
    d3 = deal_text('d:xunlian.xls')
    d4 = deal_text('d:test_data.xls')
    words1 = calculate_tfidf(d1)
    for t in words1:
        print t
    words2 = calculate_tfidfN(d1)
    words3 = calculate_tfidf(d2)
    words4 = calculate_tfidfN(d2)
    train_data = get_data2(d3)
    print train_data
    calculate_accuracy(get_data2(d0))
    print train_data#$$
def Predict(d0):
    predictors = ["age","sex", "Xiongbi", "Gutou", "Zhiqiguan", "Wenli", "Ye","Feimen","Ying","Jiejie","Guang","Gai"]
    leaf_size = 1
    n_estimators_size = 800
    alg = RandomForestClassifier(min_samples_leaf=leaf_size, n_estimators=n_estimators_size, random_state=50)
    alg.fit(train_data[predictors], train_data['Bing'])
    predict = alg.predict(d0[predictors])
    return predict
if __name__ == '__main__':
    d0 = deal_text('d:zong.xls')
    d1 = deal_text('d:yin.xls')
    d2 = deal_text('d:yang.xls')
    d3 = deal_text('d:xunlian.xls')
    d4 = deal_text('d:test_data.xls')
    words1 = calculate_tfidf(d1)
    for t in words1:
        print t
    words2 = calculate_tfidfN(d1)
    words3 = calculate_tfidf(d2)
    words4 = calculate_tfidfN(d2)
    train_data = get_data2(d3)
    calculate_accuracy(get_data2(d0))  ##测试
    aa = Predict(get_data_test(d4))   ##预测
    bb = [] 
    count_a = 0
    for i in aa:
        if i == 0:
            bb.append(u'阳')
        else:
            bb.append(u'阴')
        count_a+=1
    bb
    d4[u'阳性'] = bb
    d4.to_excel('d:predict.xls')