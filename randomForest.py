# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 22:39:40 2020

@author: Maria
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Importing the dataset
dataset = pd.read_csv('datasets/balanced_Dataset/export_dataframe.csv')
X = dataset.iloc[:, :-2].values # all rows, all columns except last result and 3 months answer - (1198, 39)
y_before = dataset.iloc[:, 39].values # all rows, last column (result) keep a record to compare later

# Encoding categorical data
# encoding the output
onehotencoder = OneHotEncoder(categorical_features = [39])
y = onehotencoder.fit_transform(dataset.values).toarray()
y = y[:, 0:4]
# encoding the input
onehotencoder = OneHotEncoder(categorical_features = [6, 7, 14, 21, 36]) 
# etiology, portal thrombosis, pretransplant status performance, cause of death, cold ischemia time 
X = onehotencoder.fit_transform(X).toarray()

# Separating each column to predict separate classes
y_1 = y[:, 0]
y_2 = y[:, 1]
y_3 = y[:, 2]
y_4 = y[:, 3]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y_1, test_size = 0.2, random_state = 0)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
'''n_estimators: number of trees in forest, which will predict if each
   match is a good match or not, based on average predictions, the result
   will be chosen. Default is 10. Try different numbers. Detect overfitting.
   criterion: entropy measures the quality of the split
'''

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




