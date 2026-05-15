# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:35:49 2026

@author: ilknur
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('Churn_Modelling.csv')



X= veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values
#encoder: kategorik -> numeric
from sklearn import preprocessing
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough"
                        )
X = ohe.fit_transform(X)
X = X[:,1:]#ilk sütunu siliyoruz


    
#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)



# Yapay Sinir Ağları
import keras

from keras.models import Sequential
from keras.layers import Dense, Input

classifier = Sequential() #yapay sinir ağı oluştu

# 1. Giriş katmanını (11 nitelik/özellik) ayrı olarak tanımlıyoruz:
classifier.add(Input(shape=(11,)))

# 2. İlk gizli katman (init yerine kernel_initializer kullandık):
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
#İKİNCİ Gizli Katman (Örn: Bunu da 6 veya 8 nöronlu yapabilirsin)
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
#çıkış katmanı
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, epochs=50)
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)

print(cm)













