# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:59:54 2026

@author: ilknur
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('sepet.csv', header = None, sep=";")

t =[]
for i in range(0,7501):
    t.append([str(veriler.values[i,j]) for j in range (0,20)])

from apyori import apriori
kurallar = apriori(t,min_support=0.01, min_confidence=0.2, min_lift =3, min_length =2)
print(list(kurallar))