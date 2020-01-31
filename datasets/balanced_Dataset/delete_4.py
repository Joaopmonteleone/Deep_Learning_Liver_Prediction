# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 18:49:26 2020

@author: Maria
"""

import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('datasets/openrefine/training1200_ordinal_output.csv')

hello = []

count = 0
for index, row in dataset.iterrows():
   if row[39] == 4:
      hello.append(count)
   count += 1
   
# leave 60 observations with output 4
hello = hello[:len(hello)-60]
      
for i in hello:
   print(i)
   dataset = dataset.drop([i])
   
export_csv = dataset.to_csv (r'C:\Users\Maria\Desktop\export_dataframe.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
