# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:39:06 2026

@author: ilknur
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# veri yukleme
veriler = pd.read_csv('maaslar.csv', sep=";")

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()

print('Linear R2 değeri')
print(r2_score(Y , lin_reg.predict(X)))
