# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 00:18:07 2026

@author: ilknur
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm

# veri yukleme
veriler = pd.read_csv('maaslar_yeni.csv', sep=";")

x = veriler.iloc[:,2:5] #bağımlı ve bağımsız değişkeni ayırıyoruz
y = veriler.iloc[:,5:]#ve değişkenelere atıyoruz
X = x.values
Y = y.values

#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)



#p value hesaplama
import statsmodels.api as sm
model =sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())

print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


#tahminler


print('poly ols')
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X),X))
print(model2.fit().summary())

print('Polynomial R2 değeri')
print(r2_score(Y , lin_reg2.predict(poly_reg.fit_transform(X))))

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))






from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)



print('svr ols')
model3= sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())

print('SVR R2 değeri')
print(r2_score(y_olcekli ,svr_reg.predict(X)))



#decision tree regression
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)


print('decision tree ols')
model4= sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())

print('decision tree R2 değeri')
print(r2_score(Y , r_dt.predict(X)))


#random forest regresyonu
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)# estimators kaç tane alt ağaç oluşturacağını belirliyor
rf_reg.fit(X,Y.ravel())



print('random forest ols')
model5= sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())


print('random forest R2 değeri')
print(r2_score(Y,rf_reg.predict(X))) #doğru değerin tahmin değeriyle karşılaştırılması




#Ozet R2 değerleri
print('-----------------------')
print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))

print('Polynomial R2 degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

print('SVR R2 degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


print('Decision Tree R2 degeri')
print(r2_score(Y, r_dt.predict(X)))

print('Random Forest R2 degeri')
print(r2_score(Y, rf_reg.predict(X)))

