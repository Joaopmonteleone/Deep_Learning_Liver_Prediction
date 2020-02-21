# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:16:40 2020

@author: Maria
"""

import pandas as pd
dataset = pd.read_csv('datasets/regAll.csv')

X_before = dataset.iloc[:, :-1] # all rows, all columns except last result and 3 months answer - (1198, 39)
y_before = dataset.iloc[:, (dataset.values.shape[1]-1)].values # all rows, last column (result) keep a record to compare later

import numpy as np

age = pd.DataFrame(np.random.randint(15,77,size=(1000, 1)), columns=list('a'))
age = age.rename(columns={'a':'age'})

global concatenated

columnNames = []
for col in X_before.columns:
    columnNames.append(col)
columnNames.remove('age')

for col in columnNames:
    df = pd.DataFrame(np.random.randint((X_before[col].min()),(X_before[col].max()+1),size=(1000, 1)), columns=list('a'))
    df = df.rename(columns={'a':col})
    concatenated = pd.concat([age, df], axis=1)


gender = pd.DataFrame(np.random.randint(0,2,size=(1000, 1)), columns=list('a'))
gender = gender.rename(columns={'a':'gender'})

bmibasal = pd.DataFrame(np.random.randint(14,70,size=(1000, 1)), columns=list('a'))
bmibasal = bmibasal.rename(columns={'a':'bmibasal'})

diabetesPreTx = pd.DataFrame(np.random.randint(0,2,size=(1000, 1)), columns=list('a'))
diabetesPreTx = diabetesPreTx.rename(columns={'a':'diabetesPreTx'})

gender = pd.DataFrame(np.random.randint(0,2,size=(1000, 1)), columns=list('a'))
gender = gender.rename(columns={'a':'gender'})

gender = pd.DataFrame(np.random.randint(0,2,size=(1000, 1)), columns=list('a'))
gender = gender.rename(columns={'a':'gender'})

gender = pd.DataFrame(np.random.randint(0,2,size=(1000, 1)), columns=list('a'))
gender = gender.rename(columns={'a':'gender'})

concatenated = pd.concat([age, gender, bmibasal,
                          diabetesPreTx], 
                          axis=1)

X_before = X_before.append(concatenated, ignore_index = True, sort=False)