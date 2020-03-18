# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:45:45 2020

@author: Maria

Script to only have 1 class, survived or not after 3 months
"""

import pandas as pd

# Importing the dataset
dataset = pd.read_csv('claAll.csv')

ones = []
twos = []

count = 0
for index, row in dataset.iterrows():
   if row[38] == 1:
      ones.append(count)
   else:
       twos.append(count)
   count += 1
   
for i in ones:
    dataset.iloc[i, 38] = 0
for i in twos:
    dataset.iloc[i, 38] = 1

export_csv = dataset.to_csv(r'C:\Users\Maria\Desktop\Deep_Learning_Liver_Prediction\datasets\cla3monthSurvival.csv',
                             index = None, 
                             header=True) 
    