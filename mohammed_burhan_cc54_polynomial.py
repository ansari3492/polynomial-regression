# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 10:29:54 2018

@author: Lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("bluegills.csv")
features=data["age"].values.reshape(-1,1)
labels=data["length"].values

#for polynomial data
from sklearn.preprocessing import PolynomialFeatures
pol=PolynomialFeatures(degree = 5)
features_pol=pol.fit_transform(features.reshape(-1,1))

#for linear regression
from sklearn.linear_model import LinearRegression
linear=LinearRegression()
linear.fit(features_pol,labels)
pred=linear.predict(5)

#visual the polynomial results
plt.scatter(features,labels,color='CMY')
plt.plot(features,linear.predict(pol.fit_transform(features.reshape(-1,1))),color='blue')
plt.title("polynomial(polynomial regression)")
plt.xlabel("age")
plt.ylabel("length")
plt.show()

features_grid=np.arange(min(features),max(features),0.1)
features_grid=features_grid.reshape(-1,1)
plt.scatter(features,labels,color='CMY')
plt.plot(features_grid,linear.predict(pol.fit_transform(features_grid)),color = 'blue')
plt.title("age vs length(polynomial regression)")
plt.xlabel("age")
plt.ylabel("length")
plt.show()

