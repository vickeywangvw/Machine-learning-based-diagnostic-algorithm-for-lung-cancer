#数据处理
import pandas as pd
import numpy as np
#数据可视化
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot") #用R语言中的ggplot模式，做出的图会更加好看
plt.rcParams['font.family'] = ['simhei'] #让图表可以显示中文
#scikitlearn包引入
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.learning_curve import learning_curve  #learning_curve有助于我们分析数据的欠拟合／过拟合情况
#文本分词
import jieba

def result(x):
    if x == "阳性":
        return 0
    if x == "阴性":
        return 1

def age_process(x):
    if x.find("岁"):
        position1 = x.find("岁")
        x = x[:position1]
        x = int(x)
    elif x.find("天"):
        position2 = x.find("天")
        x = x[:position2]
        x = int(x/356)
    elif x.find("月"):
        position3 = x.find("月")
        x = x[:position2]
        x = int(x/12)
    return x

def method(x):
    x = x.replace('[','').replace(']','').replace(',','').replace('胸部','').replace("+",'')
    return x

def replace1(x):
    x = str(x)
    x = x.replace("(","").replace(")","")
    x = float(x)
    return x

def change(x):
    if x == 0:
        return "阳性"
    else:
        return "阴性"




def main():
    data = pd.read_csv("/users/wangkaixi/desktop/test.csv")
    data.columns = ['type','age','sex','checktime','number','facilities','method','description','diagnose','advise','positive','hurry']
    #data.columns = ['type','age','sex','checktime','number','facilities','method','description','diagnose','advise','positive','hurry']
    data.positive = data.positive.apply(lambda x : result(x))
    data.age = data.age.apply(lambda x : age_process(x))
    data = data.drop(['checktime','number','facilities','advise'],axis = 1)
    data['method'] = data.method.apply(lambda x:method(x))
    data_type = pd.get_dummies(data.type,prefix = 'type')
    data_sex = pd.get_dummies(data.sex,prefix = 'sex')
    data_method = pd.get_dummies(data.method,prefix = 'method')
    #data_train_diagnose = pd.get_dummies(data_train.diagnose,prefix = 'diagnose')
    data_hurry = pd.get_dummies(data.hurry,prefix = 'hurry')
    scaler = preprocessing.StandardScaler()
    age_scaler = scaler.fit(data.age)
    data['age'] = scaler.fit_transform(data['age'],age_scaler)
    data['age'] = pd.qcut(data['age'],4)
    data_age = pd.get_dummies(data.age,prefix = 'age')
    #concat
    data = pd.concat([data,data_type,data_sex,data_method,data_hurry,data_age],axis = 1).drop(['type','sex','method','diagnose','hurry','age','description'],axis = 1)
    data = data.drop(["type_体检",'method_平扫定位'],axis = 1)
    df1 = pd.read_csv("/users/wangkaixi/desktop/trainnum.csv")
    df2 = pd.read_csv("/users/wangkaixi/desktop/testnum.csv")
    df1 = df1.head(14999)
    df1.columns = ["num"]
    df2 = df2.head(5000)
    df2.columns = ["num"]
    df_num = pd.concat([df1,df2],axis = 0)
    df_num.index = np.arange(0,19999)
    data = pd.concat([data,df_num],axis = 1)
    data.num = data.num.apply(lambda x : replace1(x))

    return data

def main1():
    data = pd.read_csv("/users/wangkaixi/desktop/data_test1.csv")
    #data.columns = ['type','age','sex','checktime','number','facilities','method','description','diagnose','advise','positive','hurry']
    #data.columns = ['type','age','sex','checktime','number','facilities','method','description','diagnose','advise','positive','hurry']
    data.positive = data.positive.apply(lambda x : result(x))
    data.age = data.age.apply(lambda x : age_process(x))
    data = data.drop(['checktime','number','facilities','advise'],axis = 1)
    data['method'] = data.method.apply(lambda x:method(x))
    data_type = pd.get_dummies(data.type,prefix = 'type')
    data_sex = pd.get_dummies(data.sex,prefix = 'sex')
    data_method = pd.get_dummies(data.method,prefix = 'method')
    #data_train_diagnose = pd.get_dummies(data_train.diagnose,prefix = 'diagnose')
    data_hurry = pd.get_dummies(data.hurry,prefix = 'hurry')
    scaler = preprocessing.StandardScaler()
    age_scaler = scaler.fit(data.age)
    data['age'] = scaler.fit_transform(data['age'],age_scaler)
    data['age'] = pd.qcut(data['age'],4)
    data_age = pd.get_dummies(data.age,prefix = 'age')
    #concat
    data = pd.concat([data,data_type,data_sex,data_method,data_hurry,data_age],axis = 1).drop(['type','sex','method','diagnose','hurry','age','description','Unnamed: 0','positive'],axis = 1)
    #data = data_train.drop("type_体检",axis = 1)
    df3 = pd.read_csv("/users/wangkaixi/desktop/finaltestnum.csv")
    df3.columns = ['num']
    df3.num = df3.num.apply(lambda x : replace1(x))
    data = pd.concat([data,df3],axis = 1)
    
    return data

if __name__ == "__main__":
    data = main()
    data1 = main1()
    data_matrix = data.as_matrix()
    X = data_matrix[:,1:]
    y = data_matrix[:,0]
    logistic = LogisticRegression()
    logistic.fit(X,y)
    svm = LinearSVC()
    svm.fit(X,y)
    predictions = svm.predict(data1)
    data = pd.read_csv("/users/wangkaixi/desktop/data_test1.csv")
    data.positive = predictions
    data.positive = data.positive.apply(lambda x:change(x))
    #data.to_csv(path)
    

