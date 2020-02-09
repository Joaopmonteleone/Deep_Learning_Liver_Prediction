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
dataset = pd.read_csv('datasets/balanced/regressionBalanced.csv')
X = dataset.iloc[:, :-1].values # all rows, all columns except last result and 3 months answer - (1198, 39)
y_before = dataset.iloc[:, 38].values # all rows, last column (result) keep a record to compare later

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
# output NOT encoded
y = y_before
# encoding the output
onehotencoder = OneHotEncoder(categorical_features = [39])
y = onehotencoder.fit_transform(dataset.values).toarray()
y = y[:, 0:4]
# encoding the input
onehotencoder = OneHotEncoder(categorical_features = [6, 7, 14, 21, 36]) 
X = onehotencoder.fit_transform(X).toarray()
# etiology, portal thrombosis, pretransplant status performance, cause of death, cold ischemia time 

# Separating each column to predict separate classes
y_1 = y[:, 0]
y_2 = y[:, 1]
y_3 = y[:, 2]
y_4 = y[:, 3]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_4, test_size = 0.2, random_state = 0)
''' test_size is 20% = 0.2
    random_state is a generator for random sampling '''

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

X_train_scaled = sc_X.fit_transform(X_train)


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
    
from randomForest import randomForest, plotRandomForest, makeTree, getImportance
rf, predictions = randomForest(X_train, X_test, y_train, y_test)
plotRandomForest(y_test, predictions)
feature_list = makeTree(rf, X_train, X_train_scaled, y_train)  
feature_importances = getImportance(rf, feature_list)
    
    
from randomForest import randomForest
rfModel = randomForest(X_train, y_train, X_test, y_test)
results = ['Random Forest', str(rfModel.getMAE()), 
           str(rfModel.getMSE()), str(rfModel.getMAPE())]
# Get top 4 instances
importances = rfModel.getImportance()[:4]
for i, j in importances:
     list = ['', '', '', '', '', '', i, j]
     results.append(list)
    
    
