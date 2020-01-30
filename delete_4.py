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
      
for i in hello:
   dataset = dataset.drop([hello[i]])
   