#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 14:50:11 2025

@author: emirhanerdem
"""
#import library
import pandas as pd
import matplotlib.pyplot as plt

#import data
df = pd.read_csv("regresyon_veri_seti.csv",sep=";")

#plot data
plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

#linear regression 

#sklearn library
from sklearn.linear_model import LinearRegression

#linear regression model
linear_reg=LinearRegression()
x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)
 
linear_reg.fit(x,y)

#prediction
import numpy as np

b0 = linear_reg.predict([[0]]) #b0 değeri yani y yi kestiği nokta
#b0 = linear_reg.intercept_ yukarıdakiyle aynı amaç

b1 = linear_reg.coef_ #eğimi bulduk
 
#maas= 1663 + 1138*deneyim 
#maas_yeni = 1663 + 1138*11 #11yıllık deneyimi olan kişinin maaşını tespit etme
#linear_reg.predict([[11]])

# visualize line
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1) #deneyim
y_head=linear_reg.predict(array) #maas

plt.figure(figsize=(8,10))
#plt.subplot(2,1,2)
plt.scatter(x, y)
plt.plot(array, y_head, color="green")
plt.xlabel("deneyim")
plt.ylabel("maaş")
plt.title("Linear Regression Fit")

plt.tight_layout()
plt.show()



#linear_reg.predict([100])
