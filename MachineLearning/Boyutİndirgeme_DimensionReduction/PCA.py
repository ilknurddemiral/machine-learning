# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:32:13 2026

@author: ilknur
"""

#kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veri kümesi
veriler = pd.read_csv('Wine.csv', sep =";")
x = veriler.iloc[:, 0:13].values
y = veriler.iloc[:, 13].values 

#eğitim ve test kümelelerinin bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)

#ölçeklendirme
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

x_train2 = pca.fit_transform(x_train)
x_test2 = pca.transform(x_test) #aynı dönüşüm test verisine uygulanır

#pca dönüşümünden önce gelen LR
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

#pca dönüşümden sonra gelen LR
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(x_train2, y_train)

#tahminler
y_pred = classifier.predict(x_test)
y_pred2 = classifier2.predict(x_test2)

from sklearn.metrics import confusion_matrix
#actual / PCA olmadan çıkan sonuç
print('gercek / PCAsiz')
cm = confusion_matrix(y_test, y_pred)
print(cm)


#actual / PCA sonrası çıkan sonuç
print('gercek / PCA ile')
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

#PCA sonrası / PCA öncesi
print('PCA siz / PCA li')
cm3 = confusion_matrix(y_pred, y_pred2)
print(cm3)















