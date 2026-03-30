# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:12:02 2026

@author: ilknur
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


veriler = pd.read_csv('veriler.csv', sep=";")
print(veriler)


x= veriler.iloc[:,1:4].values #bağımsız değişken
y = veriler.iloc[:,4:].values#bağımlı değişken
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#decision tree (entropi ile) 
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier( criterion='entropy' )

dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)





