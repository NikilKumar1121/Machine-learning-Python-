#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


bank = pd.read_csv(r"C:\Users\NIKIL\Desktop\R codes\logistic\bank-full(2).csv",header=0,encoding='unicode_escape')
bank.head()
print(bank.dtypes)
sns.countplot(bank['y']);
sns.countplot(bank['marital']);
sns.countplot(bank['education']);
sns.countplot(bank['default']);
sns.countplot(bank['housing']);
bank.columns
bank.isnull().sum()
sns.pairplot(bank)
bank.head()
# creating dummy 
dummy=pd.get_dummies(bank,drop_first=True)
dummy.columns
dummy.head()
dummy.info()
x=dummy.iloc[:,:42]
x.head()
y=dummy.iloc[:,42]
y.head()
#splitting data to test and train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)

#logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
model.coef_
model.score(x_train,y_train)
model.score(x_test,y_test)
ypred=model.predict(x_test)


from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score
confusion_matrix(y_test,ypred)
accuracy_score(y_test,ypred)
ypred_prob = model.predict_proba(x_test)[::,1]

#ROC curve
fpr,tpr,_=roc_curve(y_test,ypred_prob)
roc_auc_score(y_test,ypred_prob)
plt.plot(fpr,tpr,label="roc curve")
plt.legend(loc=4)



