# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:16:40 2020

@author: Maria
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

dataset = pd.read_csv('datasets/regAll.csv')

###############################################
#                  Random                     #
###############################################
# make a dataframe of age so the other columns can be added later on
age = pd.DataFrame(np.random.randint(15,77,size=(8000, 1)), columns=list('a'))
age = age.rename(columns={'a':'age'})
global concatenated
# make a list of all column names and remove 'age' from it
columnNames = []
for col in dataset.columns:
    columnNames.append(col)
columnNames.remove('age')
# for every column, create 8000 rows of random numbers between that column's max and min values
for col in columnNames:
    df = pd.DataFrame(np.random.randint((dataset[col].min()),(dataset[col].max()+1),size=(8000, 1)), columns=list('a'))
    df = df.rename(columns={'a':col}) # rename column name to what it is
    age = pd.concat([age, df], axis=1) # join lists
dataset = dataset.append(age, ignore_index = True, sort=False) # add synthetic data to original data
dataset.to_csv(r'C:\Users\Maria\Desktop\Deep_Learning_Liver_Prediction\datasets\syntheticDataReg.csv',
               index = False)


###############################################
#          sklearn.make_regression            #
###############################################
X,y = make_regression(n_samples=10, n_features=38, n_informative=10, n_targets=1, noise=0.1, random_state=7)
