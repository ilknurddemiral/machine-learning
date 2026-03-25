# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 19:56:56 2026

@author: ilknur

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('eksikveriler.csv', sep=";")
print(veriler)

veriler = pd.read_csv('veriler.csv', sep=';', skip_blank_lines=True)
veriler = veriler.dropna(how='all')
print(veriler.tail())




c=veriler.iloc[:,-1:].values
print(c)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
c[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(c)

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)






boy = veriler[['boy']]
print(boy)

veriler.iloc[:,1:4] = (
    veriler.iloc[:,1:4]
    .replace(',', '.', regex=True)
    .astype(float)
)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy ='mean')
Yas =veriler.iloc[:,1:4].values
print(Yas)
print("///////")

imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)




ulke =veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

print(ohe.categories_)

print(list(range(22))) #sıfırdan 22 e kadar sayıları yazar
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)

sonuc2= pd.DataFrame(data=Yas, index= range(22), columns =['boy','kilo','yas'])

cinsiyet = veriler.iloc[:, -1].values
print(cinsiyet)

sonuc3 =pd.DataFrame(data =c[:,:1], index = range(22), columns = ['cinsiyet'])
print(sonuc3)


s= pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2 = pd.concat([s,sonuc3], axis=1)
print(s2)


#birleştirme işleminden sonra bölme işlemine geçiyoruz
#boy kilo ve ülkeye göre cinsiyet tahmini
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test =train_test_split(s, sonuc3,test_size=0.33, random_state=0)

#verilerin olceklenmesi. Verileri birbirine yakın sayılara dönüşütür
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


boy = s2.iloc[:,3:4].values
print(boy)

sol = s2.iloc[:,:3]
sag =s2.iloc[:,4:]
veri= pd.concat([sol,sag],axis=1)

x_train, x_test, y_train, y_test =train_test_split(veri, boy,test_size=0.33, random_state=0)

r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)







import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int), values=veri, axis=1)

X_l=veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())











