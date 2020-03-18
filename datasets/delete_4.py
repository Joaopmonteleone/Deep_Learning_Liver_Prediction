# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 18:49:26 2020

@author: Maria
"""

import pandas as pd

# Importing the dataset
dataset = pd.read_csv('classification/cla3monthSurvival.csv')

hello = []

count = 0
for index, row in dataset.iterrows():
   if row[38] == 0:
      hello.append(count)
   count += 1
   
   
for i in hello:
   dataset = dataset.drop([i])
   
export_csv = dataset.to_csv(r'C:\Users\Maria\Desktop\Deep_Learning_Liver_Prediction\datasets\classification\claOnly1.csv',
                             index = None, 
                             header=True) 
