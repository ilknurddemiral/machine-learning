# -*- coding: utf-8 -*-
"""


@author: ilknur
"""


#kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#kodlar
#veri yukleme

veriler = pd.read_csv('eksikveriler.csv')
#pd.read_csv("veriler.csv")

print(veriler)

# veri tablosunda boş satır varsa almasın
veriler = pd.read_csv('veriler.csv', sep=';', skip_blank_lines=True)
veriler = veriler.dropna(how='all')
print(veriler.tail())


#veri on isleme

boy = veriler[['boy']]
print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)

x = 10

class insan:
    boy = 180
    def kosmak(self,b):
        return b + 10

ali = insan()
print(ali.boy)
print(ali.kosmak(90))


#eksik veriler
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)



#kategorik veriler
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


