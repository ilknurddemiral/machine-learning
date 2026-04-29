# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:51:27 2026

@author: ilknur
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
veriler = pd.read_csv('Ads_CTR_Optimisation.csv', sep=";")

N = 10000 # 10.000 tıklama
d = 10 # toplam 10 ilan var
oduller = [0] * d # ilk başta butun ilanların odulu 0
toplam = 0 # toplam odul
secilenler = []
birler = [0] * d
sifirlar = [0] * d
for n in range(1,N):
    ad = 0 #secilen ilan
    max_th = 0
    for i in range(0,d):
        rastbeta = random.betavariate (birler[i]+1, sifirlar[i]+1)
        if rastbeta > max_th:
            max_th = rastbeta
            ad = i
    secilenler.append(ad)
    odul = veriler.values[n,ad]
    if odul == 1:
        birler[ad] = birler[ad] +1
    else:
        sifirlar[ad] = sifirlar[ad] +1
    toplam = toplam + odul 
print('Toplam Odul:')
print(toplam)

plt.hist(secilenler)
plt.show()
