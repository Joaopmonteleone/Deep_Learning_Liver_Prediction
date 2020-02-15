# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 14:05:14 2020

@author: 40011956
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu', input_dim = 55))
    classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

def evaluate(X_train, y_train):
    classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
    
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
    mean = accuracies.mean()
    variance = accuracies.std()
    
    return accuracies, mean, variance
