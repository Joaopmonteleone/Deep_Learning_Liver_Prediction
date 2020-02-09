# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:14:18 2020

@author: Maria
"""

import pandas as pd
dataset = pd.read_csv('regEncoded4.csv')

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [51]) 
dataset = onehotencoder.fit_transform(dataset).toarray()
# etiology, 6
#portal thrombosis, 7, 13
#pretransplant status performance, 14
#cause of death, 21
#cold ischemia time, 36

# export to file
export_csv = pd.DataFrame(dataset).to_csv(r'C:\Users\Maria\Desktop\Deep_Learning_Liver_Prediction\datasets\regEncoded5.csv')
