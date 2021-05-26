#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 23:20:18 2020

@author: ozkan
"""

#%% import library
import pandas as pd
import numpy as np

#%% import dataset
df = pd.read_csv('Advertising.csv')
df = df.iloc[:,1:len(df)]


#%% visualization

#import seaborn as sns

#sns.jointplot(x = "TV", y = "sales", data = df, kind = 'reg')

#%% Linear Regression  -------  
from sklearn.linear_model import LinearRegression
X = df[["TV"]]

y = df[["sales"]]

reg = LinearRegression()

model = reg.fit(X, y)

b0 = model.intercept_ 

b1 = model.coef_ 

model.score(X,y)  

model.predict([[165]])

yeni_veri = [[5],[15],[30]]

print(model.predict(yeni_veri))




real_y = y[0:10]

predict_y = pd.DataFrame(model.predict(X)[0:10])

errors = pd.concat([real_y, predict_y],axis=1)

errors.columns = ['real_y', 'predict_y']

errors['error'] = errors['real_y'] - errors['predict_y']

errors['errors_square'] = errors['error']**2

print(np.mean(errors['errors_square']))