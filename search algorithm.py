
# coding: utf-8

# In[ ]:


import pandas as pd
import time
import jieba

def dataprocess(path):
    data = pd.read_csv(path)
    data.columns = ['type','age','sex','checktime','number','facilities','method','description','diagnose','advise','positive','hurry']
    data.advise = data.advise.astype("str")
    data.diagnose = data.diagnose.astype("str")
    data.number = data.number.astype("str")
    return data

def searchprocess(data):
    keywords = str(input("请输入你的关键词："))
    k = 0
    data['big'] = data.type.apply(str)+data.age.apply(str)+data.sex.apply(str)+data.checktime.apply(str)+data.number.apply(str)+data.facilities.apply(str)+data.method.apply(str)+data.description.apply(str)+data.diagnose.apply(str)+data.advise.apply(str)+data.positive.apply(str)+data.hurry.apply(str)
    newdata = data[0:1]
    while k <= 19998:
        if keywords in data.big[k]:
            new_data = data[k:k+1]
            newdata = pd.concat([newdata,new_data],axis = 0)
        k += 1
    newdata = newdata.drop(0).drop_duplicates()
    return newdata

def ratio(newdata):
    
    a = newdata.positive.value_counts()[0]/(newdata.positive.value_counts()[1] + newdata.positive.value_counts()[0])
    print(a)



    #print("阳性的比率是%d"%(newdata[newdata.positive == "阳性"].value_counts()/newdata.positive.value_counts())

if __name__ == "__main__":
    data = dataprocess("e:/test.csv")
    newdata = searchprocess(data)
    print("-------------------------------")
    time.sleep(1.5)
    print("您搜索的结果如下")
    print("-------------------------------")
    print(newdata)
    #ratio(newdata)


# In[ ]:


data.to_excel("e://result.xls")

