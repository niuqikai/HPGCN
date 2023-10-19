#代码使用说明<br>
HPGCN算法是基于GCN模型(https://github.com/tkipf/gcn),TCMSP,ETCM数据等进行中药药性（寒热）预测的算法模型.具体使用方法如下：

##1 特征选择<br>
featureselect.py主要用于模型特征选择
其中X_new = SelectKBest(f_classif,k=len(dt.iloc[:,1:-1].columns)).fit(dt.iloc[:,1:-1] ,dt.iloc[:,-1])函数中
可选择f_classif或者chi2

##2 生成中药-中药网络和中药特征向量<br>
herb_property.py根据选择的特征，生成cora.cites和cora.content<br>
分布对应中药-中药网络和中药靶点特征矩阵<br>

##3 train.py根据生成训练集和测试集结果<br>
分别记录混淆矩阵和acc,recall,precision,F1,auc值<br>

##4 LR_XGBT_SVM_Bayes_KNN.py生成LR、XGBT等模型的训练集和测试集结果<br>


仅供科学研究使用，请勿用作商业用途。<br>
联系方式：niuqikai@qq.com<br>
