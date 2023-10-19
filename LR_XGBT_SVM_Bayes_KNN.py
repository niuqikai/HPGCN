from sklearn.model_selection  import train_test_split,KFold
import pandas as pd
from sklearn import metrics
from pandas.core.frame import DataFrame
import numpy as np
from sklearn.metrics import confusion_matrix,roc_auc_score

import xgboost as xgb
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC #使用支持向量机算法
from sklearn.tree import DecisionTreeClassifier

#import matplotlib.pyplot as plt
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import seaborn as sns
# data = data.fillna(-1)
filepath = 'herb_class/'
filelabel = 'cora.content'


def Evaluating_Indicator(y_true,y_pred):
    '''
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    acc = (TP + TN)/(TN + FP + FN + TP)
    recall = TP/(TP + FN)
    precision = TP/(TP + FP)
    F1 = 2/(1.0/precision + 1.0/recall)
    auc = roc_auc_score(y_true,y_pred)
    '''
    print((confusion_matrix(y_true, y_pred).ravel()))
    acc = metrics.accuracy_score(y_true,y_pred)
    recall = metrics.recall_score(y_true,y_pred)
    precision = metrics.precision_score(y_true,y_pred)
    F1 = metrics.f1_score(y_true,y_pred)
    auc = metrics.roc_auc_score(y_true,y_pred)
    return acc,recall,precision,F1,auc

def write_result_to_file(acc,recall,precision,F1,auc):
    with open('rs.csv','a') as fw:
        fw.write(str(acc))
        fw.write(',')
        fw.write(str(recall))
        fw.write(',')
        fw.write(str(precision))
        fw.write(',')
        fw.write(str(F1))
        fw.write(',')
        fw.write(str(auc))
        fw.write('\n')

df = pd.read_csv(filepath+filelabel,header=None,encoding='gbk',sep='\t')
df = df.replace('热',1)
df = df.replace('寒',0)
#df = df[df['loan_mon_cnt']>7]
#x = df.iloc[:, [10,11,12,13,14,15,16]].values
x = df.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]].values

y = df.iloc[:, 31].values


#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) # 为了看模型在没有见过数据集上的表现，随机拿出数据集中30%的部分做测试
kf = KFold(n_splits=5,shuffle=False)
for train_index , test_index in kf.split(df):  # 调用split方法切分数据
    X_train = x[train_index]
    y_train = y[train_index]
    X_test = x[test_index]
    y_test = y[test_index]
    print(train_index)
    #print(X_train,y_train,X_test,y_test)
    dtrain = xgb.DMatrix(data = X_train,label = y_train)
    dtest = xgb.DMatrix(data = X_test, label = y_test)

    param = {'max_depth':3, 'eta':0.075, 'silent':1,'lambda':4,'gamma':0, 'subsample':0.5,
             'consample_bytree':0.8,
             'objective':'binary:logistic', 'nthread':4, 'min_child_weight':12,'seed':2}
    evallist  = [(dtest,'eval'), (dtrain,'train')]
    num_round = 85
    bst = xgb.train(param, dtrain, num_round, evallist)

    y_pred_train = bst.predict(dtrain)
    y_pred_train = (y_pred_train >= 0.5)*1
    y_pred_test = bst.predict(dtest)
    y_pred_test = (y_pred_test >= 0.5)*1

    print('xgt:')
    acc,recall,precision,F1,auc = Evaluating_Indicator(y_train,y_pred_train)
    write_result_to_file(acc, recall, precision, F1, auc)
    acc,recall,precision,F1,auc = Evaluating_Indicator(y_test,y_pred_test)
    write_result_to_file(acc, recall, precision, F1, auc)


    #LR
    log_reg = LogisticRegression(max_iter=1000)
    #给模型喂入数据
    clm=log_reg.fit(X_train,y_train)

    #使用模型对测试集分类预测,并打印分类结果
    #print(clm.predict(X_test))
    #最后使用性能评估器，测试模型优良，用测试集对模型进行评分
    print('LR:')
    y_pred_train = clm.predict(X_train)
    y_pred_test = clm.predict(X_test)
    acc, recall, precision, F1, auc = Evaluating_Indicator(y_train, y_pred_train)
    write_result_to_file(acc, recall, precision, F1, auc)
    acc, recall, precision, F1, auc = Evaluating_Indicator(y_test, y_pred_test)
    write_result_to_file(acc, recall, precision, F1, auc)


    #贝叶斯分类
    bayes_modle = BernoulliNB()
    bys = bayes_modle.fit(X_train,y_train)
    y_pred_train = bys.predict(X_train)
    y_pred_test = bys.predict(X_test)
    print('bayes:')
    acc, recall, precision, F1, auc = Evaluating_Indicator(y_train, y_pred_train)
    write_result_to_file(acc, recall, precision, F1, auc)
    acc, recall, precision, F1, auc = Evaluating_Indicator(y_test, y_pred_test)
    write_result_to_file(acc, recall, precision, F1, auc)


    #KNN
    KNN=KNeighborsClassifier(n_neighbors=9)
    knn = KNN.fit(X_train,y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)
    #评估模型的得分
    print("KNN:")
    acc, recall, precision, F1, auc = Evaluating_Indicator(y_train, y_pred_train)
    write_result_to_file(acc, recall, precision, F1, auc)
    acc, recall, precision, F1, auc = Evaluating_Indicator(y_test, y_pred_test)
    write_result_to_file(acc, recall, precision, F1, auc)


    #SVM
    clf = SVC(kernel='rbf')
    svm = clf.fit(X_train, y_train)
    y_pred_train = svm.predict(X_train)
    y_pred_test = svm.predict(X_test)
    # 预测分类
    print("SVM")
    acc, recall, precision, F1, auc = Evaluating_Indicator(y_train, y_pred_train)
    write_result_to_file(acc, recall, precision, F1, auc)
    acc, recall, precision, F1, auc = Evaluating_Indicator(y_test, y_pred_test)
    write_result_to_file(acc, recall, precision, F1, auc)
