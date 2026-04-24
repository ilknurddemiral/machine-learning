# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:17:27 2026

@author: ilknur
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv' , sep=";")
X = veriler.iloc[:,3:].values


#Hierarchical clustering(hiyerarşik bölütleme)
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
#bir önceki satırda k değerini, hangi hesaplama metodunu kullanacağımızı, 
#ve iki farklı küme arasındaki mesafenin hesaplanacağı metodunu yazıyoruz.
Y_tahmin = ac.fit_predict(X)
print(Y_tahmin)


plt.scatter(X[Y_tahmin==0,0], X[Y_tahmin==0,1],s=100,c='red')
plt.scatter(X[Y_tahmin==1,0], X[Y_tahmin==1,1],s=100,c='blue')
plt.scatter(X[Y_tahmin==2,0], X[Y_tahmin==2,1],s=100,c='green')
plt.scatter(X[Y_tahmin==3,0], X[Y_tahmin==3,1],s=100,c='yellow')
plt.title('HC')
plt.show()



#KMeans ile karşılaştırılması
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4, init = 'k-means++')
Y_tahmin=kmeans.fit_predict(X)
print(Y_tahmin)
plt.scatter(X[Y_tahmin==0,0], X[Y_tahmin==0,1],s=100,c='red')
plt.scatter(X[Y_tahmin==1,0], X[Y_tahmin==1,1],s=100,c='blue')
plt.scatter(X[Y_tahmin==2,0], X[Y_tahmin==2,1],s=100,c='green')
plt.scatter(X[Y_tahmin==3,0], X[Y_tahmin==3,1],s=100,c='yellow')
plt.title('KMeans')
plt.show()


#dendrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()













