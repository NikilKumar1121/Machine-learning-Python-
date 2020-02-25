#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


affairs1=pd.read_csv("D:\\python assignments\\LOGISTIC REGRESSION\\affairs(1).csv")
affairs1.head(5)
affairs1.describe()


# # VISUALISATION

affairs1.info()
print(affairs1.dtypes)
sns.countplot(x="religiousness",data=affairs1)
sns.countplot(x="affairs",data=affairs1)
affairs1.columns
affairs1.isnull()
affairs1.isnull().sum()
sns.heatmap(affairs1.isnull(),yticklabels=False,cmap="viridis")
sns.heatmap(affairs1.isnull(),yticklabels=False,cbar=False)
affairs1['gender_cat'] = affairs1.gender.map({'male':1,'female':0})
affairs1.head()
affairs1['children_cat'] = affairs1.children.map({'yes':1,'no':0})
affairs1.head()
affairs1['affairs_cat'] = 0
affairs1.head()
affairs1.loc[affairs1['affairs']>0,'affairs_cat'] = 1
affairs1=affairs1.drop(['gender','children','affairs'],axis=1)
affairs1.tail()


x=affairs1.iloc[:,:8]
x.head(2)
y=affairs1.iloc[:,8]
y.tail(2)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x,y)
model.coef_
ypred=model.predict(x)
model.score(x,y)

from sklearn.metrics import accuracy_score ,confusion_matrix ,roc_curve ,roc_auc_score
accuracy_score(y,ypred)
confusion_matrix(y,ypred)
ypredprob= model.predict_proba(x)[::,1]
ypredprob

fpr,tpr,_ =roc_curve(y,ypredprob)
auc=roc_auc_score(y,ypredprob)
plt.plot(fpr,tpr,label = "roc curve")
plt.legend(loc=4)




