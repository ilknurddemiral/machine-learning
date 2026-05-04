# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:06:50 2026

@author: ilknur
"""

import numpy as np
import pandas as pd


yorumlar = pd.read_csv('Restaurant_Reviews.csv', sep=";")

import re
import nltk
durma = nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()#ekleri ayırma

from nltk.corpus import stopwords

#¶preprocessing önişleme
derlem = []
for i in range(1000):
    yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
    yorum = yorum.lower()#küçük harf yapıyoruz
    yorum = yorum.split()#kelimelere ayırıyor
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]#kelimenin kökünü bulma
    #bir önceki satırda stopword olmaan kelimeleri lsiteye atıyoruz
    yorum = ' '.join(yorum)
    derlem.append(yorum)
    
# feature extraction Öznitelik çıkarımı  
# bag of words(bow)  
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
x = cv.fit_transform(derlem).toarray()#bağımsız değişken
y = yorumlar.iloc[:,1].values #bağımlı değişken


#makine öğrenmesi machine learning
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x, y,test_size=0.20, random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)

y_pred = gnb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)






   
