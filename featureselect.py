from sklearn.feature_selection import SelectKBest,f_classif,chi2
import pandas as pd

feature_label = 'herb_class/cora.csv'
dt = pd.read_csv(feature_label,encoding='utf-8',sep=',')
#print(dt[1:-1])
print(len(dt.iloc[:,1:-1]))
print(len(dt.iloc[:,-1]))

#X_new = SelectKBest(lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y), X))), k=15).fit_transform(dt.iloc[:,1:-1] ,dt.iloc[:,-1])
X_new = SelectKBest(f_classif,k=len(dt.iloc[:,1:-1].columns)).fit(dt.iloc[:,1:-1] ,dt.iloc[:,-1])
#X_new = SelectKBest(chi2,k=len(dt.iloc[:,1:-1].columns)).fit(dt.iloc[:,1:-1] ,dt.iloc[:,-1])

df_scores = pd.DataFrame(X_new.scores_)
df_columns = pd.DataFrame(dt.iloc[:,1:-1].columns)

# 合并
df_feature_scores = pd.concat([df_columns, df_scores], axis=1)
# 定义列名
df_feature_scores.columns = ['Feature', 'Score']
# 按照score排序
df_feature_scores = df_feature_scores.sort_values(by='Score', ascending=False)
df_feature_scores.to_csv('f_class.csv')

'''
fs = list(df_feature_scores['Feature'])
cols = fs[:30]
cols.append('label')
dt[cols].to_csv('d.csv')
'''
