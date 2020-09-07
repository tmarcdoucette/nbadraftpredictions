import pandas as pd
import random
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import seaborn as sns

df = pd.read_csv("CollegeStats.csv")

df.drop(df.columns[0], axis = 1, inplace = True)
df.drop(df.columns[0], axis = 1, inplace = True)
df.drop(df.columns[1], axis = 1, inplace = True)

df['NRtg'] = df['ORtg'] - df['DRtg']

#Top School
# Yea I got skills what are you gonna do bout it
schools = ["Kentucky", "North Carolina", "Duke", "UCLA", "Kansas", "Michigan", "Indiana", "Louisville", "Syracuse", "Ohio State", "Arizona", "Michigan State", "Notre Dame", "Connecticut", "Maryland", "Georgia Tech", "Texas", "North Carolina State", "Minnesota", "St. John's (NY)"]
df['TopSchool'] = df.apply(lambda row: True if any([item in row['School'] for item in schools]) else False, axis = 1)

confs = ["Big East", "Big 12", "Big Ten", "Pac-12", "SEC", "ACC"]
df['TopConf'] = df.apply(lambda row: True if any([item in row['Conf'] for item in confs]) else False, axis = 1)

df['TopConf'] = (df['TopConf'] == True).astype(int)
df['TopSchool'] = (df['TopSchool'] == True).astype(int)

df['Class'] = df['Class'].astype('category')
df['TopConf'] = df['TopConf'].astype('category')
df['TopSchool'] = df['TopSchool'].astype('category')
df['Pos'] = df['Pos'].astype('category')

df['G'] = pd.qcut(df['G'], 5).cat.codes
df['MP'] = pd.qcut(df['MP'], 5).cat.codes
df['PER'] = pd.qcut(df['PER'], 5).cat.codes
df['TS%'] = pd.qcut(df['TS%'], 5).cat.codes
df['eFG%'] = pd.qcut(df['eFG%'], 5).cat.codes
df['ORB%']  = pd.qcut(df['ORB%'], 5).cat.codes
df['DRB%'] = pd.qcut(df['DRB%'], 5).cat.codes
df['TRB%'] = pd.qcut(df['TRB%'], 5).cat.codes
df['AST%'] = pd.qcut(df['AST%'], 5).cat.codes
df['STL%'] = pd.qcut(df['STL%'], 5).cat.codes
df['BLK%'] = pd.qcut(df['BLK%'], 5).cat.codes
df['TOV%'] = pd.qcut(df['TOV%'], 5).cat.codes
df['USG%'] = pd.qcut(df['USG%'], 5).cat.codes
df['PProd'] = pd.qcut(df['PProd'], 5).cat.codes
df['ORtg'] = pd.qcut(df['ORtg'], 5).cat.codes
df['DRtg'] = pd.qcut(df['DRtg'], 5).cat.codes
df['OWS'] = pd.qcut(df['OWS'], 5).cat.codes
df['DWS'] = pd.qcut(df['DWS'], 5).cat.codes
df['WS'] = pd.qcut(df['WS'], 5).cat.codes
df['OBPM'] = pd.qcut(df['OBPM'], 5).cat.codes
df['DBPM']  = pd.qcut(df['DBPM'], 5).cat.codes
df['NRtg'] = pd.qcut(df['NRtg'], 5).cat.codes
df['BPM'] = pd.qcut(df['BPM'], 5).cat.codes

#set seed
random.seed(111)

feature_cols_all = ['MP', 'G', 'PER', 'TS%', 'eFG%', 'TRB%', 'USG%', 'ORtg', 'DRtg', 'OWS', 'WS', 'DWS', 'TopSchool', 'TopConf', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'PProd', 'OBPM', 'DBPM', 'BPM']
feature_cols = ['MP', 'G', 'PER', 'TRB%', 'USG%', 'ORtg', 'DRtg', 'DWS', 'TopSchool']
feature_cols_topdown = ['TRB%', 'USG%', 'OWS', 'WS', 'TopSchool', 'TopConf', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'BLK%', 'USG%', 'PProd', 'OBPM', 'NRtg']

X = df[feature_cols]
y = df.Drafted

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=0)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=0)
gnb = CategoricalNB()
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

y_pred_proba = gnb.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


drafted = pd.read_csv("PredictPlayers.csv")
draftdf = pd.DataFrame.copy(drafted)

draftdf.drop(draftdf.columns[0], axis = 1, inplace = True)
draftdf.drop(draftdf.columns[0], axis = 1, inplace = True)
draftdf.drop(draftdf.columns[1], axis = 1, inplace = True)

draftdf['NRtg'] = draftdf['ORtg'] - draftdf['DRtg']

#Top School
# Yea I got skills what are you gonna do bout it
schools = ["Kentucky", "North Carolina", "Duke", "UCLA", "Kansas", "Michigan", "Indiana", "Louisville", "Syracuse", "Ohio State", "Arizona", "Michigan State", "Notre Dame", "Connecticut", "Maryland", "Georgia Tech", "Texas", "North Carolina State", "Minnesota", "St. John's (NY)"]
draftdf['TopSchool'] = draftdf.apply(lambda row: True if any([item in row['School'] for item in schools]) else False, axis = 1)

confs = ["Big East", "Big 12", "Big Ten", "Pac-12", "SEC", "ACC"]
draftdf['TopConf'] = draftdf.apply(lambda row: True if any([item in row['Conf'] for item in confs]) else False, axis = 1)

draftdf['TopConf'] = (draftdf['TopConf'] == True).astype(int)
draftdf['TopSchool'] = (draftdf['TopSchool'] == True).astype(int)

draftdf['Class'] = draftdf['Class'].astype('category')
draftdf['TopConf'] = draftdf['TopConf'].astype('category')
draftdf['TopSchool'] = draftdf['TopSchool'].astype('category')
draftdf['Pos'] = draftdf['Pos'].astype('category')

draftdf['G'] = pd.qcut(draftdf['G'], 5, duplicates = 'drop').cat.codes
draftdf['MP'] = pd.qcut(draftdf['MP'], 5).cat.codes
draftdf['PER'] = pd.qcut(draftdf['PER'], 5).cat.codes
draftdf['TS%'] = pd.qcut(draftdf['TS%'], 5).cat.codes
draftdf['eFG%'] = pd.qcut(draftdf['eFG%'], 5).cat.codes
draftdf['ORB%']  = pd.qcut(draftdf['ORB%'], 5).cat.codes
draftdf['DRB%'] = pd.qcut(draftdf['DRB%'], 5).cat.codes
draftdf['TRB%'] = pd.qcut(draftdf['TRB%'], 5).cat.codes
draftdf['AST%'] = pd.qcut(draftdf['AST%'], 5).cat.codes
draftdf['STL%'] = pd.qcut(draftdf['STL%'], 5).cat.codes
draftdf['BLK%'] = pd.qcut(draftdf['BLK%'], 5).cat.codes
draftdf['TOV%'] = pd.qcut(draftdf['TOV%'], 5).cat.codes
draftdf['USG%'] = pd.qcut(draftdf['USG%'], 5).cat.codes
draftdf['PProd'] = pd.qcut(draftdf['PProd'], 5).cat.codes
draftdf['ORtg'] = pd.qcut(draftdf['ORtg'], 5).cat.codes
draftdf['DRtg'] = pd.qcut(draftdf['DRtg'], 5).cat.codes
draftdf['OWS'] = pd.qcut(draftdf['OWS'], 5).cat.codes
draftdf['DWS'] = pd.qcut(draftdf['DWS'], 5).cat.codes
draftdf['WS'] = pd.qcut(draftdf['WS'], 5).cat.codes
draftdf['OBPM'] = pd.qcut(draftdf['OBPM'], 5).cat.codes
draftdf['DBPM']  = pd.qcut(draftdf['DBPM'], 5).cat.codes
draftdf['NRtg'] = pd.qcut(draftdf['NRtg'], 5).cat.codes
draftdf['BPM'] = pd.qcut(draftdf['BPM'], 5).cat.codes

#set seed
random.seed(111)

feature_cols_pred_all = ['MP', 'G', 'PER', 'TS%', 'eFG%', 'TRB%', 'USG%', 'ORtg', 'DRtg', 'OWS', 'WS', 'DWS', 'TopSchool', 'TopConf', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'PProd', 'OBPM', 'DBPM', 'BPM']
feature_cols_pred = ['MP', 'G', 'PER', 'TRB%', 'USG%', 'ORtg', 'DRtg', 'DWS', 'TopSchool']
feature_cols_pred_topdown = ['TRB%', 'USG%', 'OWS', 'WS', 'TopSchool', 'TopConf', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'BLK%', 'USG%', 'PProd', 'OBPM', 'NRtg']

X_pred = draftdf[feature_cols_pred]
Y_pred = gnb.predict(X_pred)
y_series = pd.Series(Y_pred)
drafted['Drafted'] = y_series

resultdf = drafted[drafted['Drafted'] == 1]
print(resultdf)

