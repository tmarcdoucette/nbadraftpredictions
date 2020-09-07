import pandas as pd
import random
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
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

df.drop(df.columns[2], axis = 1, inplace = True)
df.drop(df.columns[2], axis = 1, inplace = True)

df['TopConf'] = (df['TopConf'] == True).astype(int)
df['TopSchool'] = (df['TopSchool'] == True).astype(int)

#set seed
random.seed(111)

feature_cols_all = ['MP', 'G', 'PER', 'TS%', 'eFG%', 'TRB%', 'USG%', 'ORtg', 'DRtg', 'OWS', 'WS', 'DWS', 'TopSchool', 'TopConf', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'PProd', 'OBPM', 'DBPM', 'BPM']
feature_cols = ['MP', 'G', 'PER', 'TRB%', 'USG%', 'ORtg', 'DRtg', 'DWS', 'TopSchool']
feature_cols_topdown = ['TRB%', 'USG%', 'OWS', 'WS', 'TopSchool', 'TopConf', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'BLK%', 'USG%', 'PProd', 'OBPM', 'NRtg']

X = df[feature_cols_all]
y = df.Drafted

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=0)

# instantiate the model (using the default parameters)
logreg = LogisticRegression(max_iter=10000)

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)

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

y_pred_proba = logreg.predict_proba(X_test)[::,1]
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
schools = ["Kentucky", "North Carolina", "Duke", "UCLA", "Kansas", "Michigan", "Indiana", "Louisville", "Syracuse", "Ohio State", "Arizona", "Michigan State", "Notre Dame", "Connecticut", "Maryland", "Georgia Tech", "Texas", "North Carolina State", "Minnesota", "St. John's (NY)"]
draftdf['TopSchool'] = draftdf.apply(lambda row: True if any([item in row['School'] for item in schools]) else False, axis = 1)

confs = ["Big East", "Big 12", "Big Ten", "Pac-12", "SEC", "ACC"]
draftdf['TopConf'] = draftdf.apply(lambda row: True if any([item in row['Conf'] for item in confs]) else False, axis = 1)

draftdf.drop(draftdf.columns[2], axis = 1, inplace = True)
draftdf.drop(draftdf.columns[2], axis = 1, inplace = True)

draftdf['TopConf'] = (draftdf['TopConf'] == True).astype(int)
draftdf['TopSchool'] = (draftdf['TopSchool'] == True).astype(int)

#set seed
random.seed(111)

feature_cols_pred_all = ['MP', 'G', 'PER', 'TS%', 'eFG%', 'TRB%', 'USG%', 'ORtg', 'DRtg', 'OWS', 'WS', 'DWS', 'TopSchool', 'TopConf', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'PProd', 'OBPM', 'DBPM', 'BPM']
feature_cols_pred = ['MP', 'G', 'PER', 'TRB%', 'USG%', 'ORtg', 'DRtg', 'DWS', 'TopSchool']
feature_cols_pred_topdown = ['TRB%', 'USG%', 'OWS', 'WS', 'TopSchool', 'TopConf', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'BLK%', 'USG%', 'PProd', 'OBPM', 'NRtg']

X_pred = draftdf[feature_cols_pred_all]
Y_pred = logreg.predict(X_pred)
y_series = pd.Series(Y_pred)
drafted['Drafted'] = y_series

resultdf = drafted[drafted['Drafted'] == 1]
print(resultdf)

