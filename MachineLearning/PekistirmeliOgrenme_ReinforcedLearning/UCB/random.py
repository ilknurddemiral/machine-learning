# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:54:41 2026

@author: ilknur
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler =pd.read_csv('Ads_CTR_Optimisation.csv', sep=";")

import random
N = 10000
d = 10
toplam =0
secilenler = []
for n in range(0,N):
    ad = random.randrange(d)#rastgele bir sayı üret
    secilenler.append(ad)
    odul = veriler.values[n,ad] #verilerdeki n. satır = 1 ise odul artıyor
    toplam = toplam + odul
    
plt.hist(secilenler)
plt.show()    