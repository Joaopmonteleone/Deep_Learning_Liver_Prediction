# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:32:59 2020

@author: 40011956
"""
###############################################
#             Data Preprocessing              #
###############################################

# Importing the dataset
import pandas as pd
dataset = pd.read_csv('datasets/balanced/regEncodedBalanced.csv')
X_before = dataset.iloc[:, :-1] # all rows, all columns except last result and 3 months answer - (1198, 39)
y_before = dataset.iloc[:, 55] # all rows, last column (result) keep a record to compare later

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
y = y_before # Do not encode for regression
# encoding the output FOR CLASSIFICATION
onehotencoder = OneHotEncoder(categorical_features = [38])
y = onehotencoder.fit_transform(dataset.values).toarray()
y = y[:, 0:4]
# encoding the input
onehotencoder = OneHotEncoder(categorical_features = [6, 7, 14, 21, 36]) 
X = onehotencoder.fit_transform(X_before).toarray()
# etiology, portal thrombosis, pretransplant status performance, cause of death, cold ischemia time 

# Separating each column to predict separate classes
y_1 = y[:, 0]
y_2 = y[:, 1]
y_3 = y[:, 2]
y_4 = y[:, 3]

# NO ENCODING
X = X_before
y = y_before

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


###############################################
#                     ANN                     #
###############################################

from ANN import neuralNetwork, predict, evaluateANN, grid_search
activation_hidden = 'relu'
activation_output = 'sigmoid' #softmax for 4, sigmoid for binary
optimizer = 'adagrad' # adagrad, adam, rmsprop, sgd
loss = 'binary_crossentropy' # binary or categorical
batch_size = 10
epochs = 500
classifier = neuralNetwork(X_train, y_train, activation_hidden, activation_output, optimizer, loss, batch_size, epochs)
  
y_pred, y_bool, cm = predict(classifier, X_test, y_test)

# doesn't work
classifier, accuracies, mean, variance = evaluateANN(X_train, y_train, activation_hidden, activation_output, optimizer, loss, batch_size, epochs)

best_parameters, best_accuracy = grid_search(X_train, y_train)


###############################################
#                Random Forest                #
###############################################
    
from randomForest import randomForest
rfModel = randomForest(X_train, y_train, X_test, y_test, X_before)
# Get top 15 instances
importances = rfModel.getImportance()
# Plot graph
randomForest.plotRandomForest(y_test, rfModel.predictions)
randomForest.makeTree(rfModel)
    
