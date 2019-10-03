#!/usr/bin/env python
# coding: utf-8

# In[3]:



import sys
from os import system
# system('cls')
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
df=pd.read_csv('kidney_disease.csv')
df['classification'][37]='ckd'
#df['classification'][37]

sys.stdout = open("output", "w")
# In[4]:


print(df.isnull().sum())
def handle_non_numeric_data(df):
    columns=df.columns.values
    
    for column in columns:
        text={}
        def convert_to_int(value):
            return text[value]
        if df[column].dtype!=np.int and df[column].dtype!=np.float:
            contents=df[column].values.tolist()
            unique_elements=set(contents)
            x=0
            for unique in unique_elements:
                if unique not in text:
                    text[unique]=x
                    x+=1
            df[column]=list(map(convert_to_int,df[column]))
            
    return df

df=handle_non_numeric_data(df)
print(df)


# In[204]:


df.drop(['id'],1,inplace=True)
for i in df:
    df[i].fillna(np.mean(df[i]) , inplace=True)
print(df.isnull().sum())


# In[205]:


# df


# In[206]:


X=df.iloc[0:,0:23].values
y=df.iloc[0:,24].values

print (X)
print(y)


# # knn

# In[207]:


X_train ,X_test,y_train, y_test=model_selection.train_test_split(X,y,test_size=0.2)
clf=neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
print("knn:", accuracy)


# # naive

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
clf= GaussianNB()
clf.fit(X_train,y_train)
print("naive:", clf.score(X_test,y_test))
print(cross_val_score(clf, X_test, y_test, cv=10, scoring='accuracy').mean())

# # decision tree

X_train ,X_test,y_train, y_test=model_selection.train_test_split(X,y,test_size=0.2)
clf=DecisionTreeRegressor()
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
print("decision tree:", accuracy)

# In[ ]:




