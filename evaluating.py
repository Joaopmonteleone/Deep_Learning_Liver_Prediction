# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:04:53 2020

@author: Maria
"""
from algorithms import importDataset
from sklearn.metrics import r2_score
import csv

results = [['dataset', 'variance score', 'max error', 'mae', 'mse', 'r2 score']]

datasets = ['regAll', 'regBalanced', 'regEncoded',
            'regEncodedBalanced', 'regNo365', 'regOnly365', 
            'regSynthetic', 'regSyntheticWith365'
           ]

def evaluateANN():
    
    for data in datasets:
        #Import the Dataset and separate X and y
        data_to_test = data + '.csv'
        X_before, y_before = importDataset(data_to_test)
        
        count = 0
        avg_explained_variance_score = 0
        avg_max_error = 0
        avg_mae = 0
        avg_mse = 0
        avg_r2_score = 0
       
        # prepare cross validation
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=3, shuffle=True)
        
        for train, test in kfold.split(X_before):
            print("Test:", count+1, " for", data_to_test)
            X_train, X_test = X_before.iloc[train], X_before.iloc[test]
            y_train, y_true = y_before[train], y_before[test]
            
            #feature scaling
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # run ANN
            from regressionAnalysis import sequentialNN
            regressor = sequentialNN(X_train, y_train, X_test, y_true)
            exp_variance_score, max_error, loss, mae, mse = regressor.getEvaluationMetrics()
            
            # get metrics
            avg_explained_variance_score += exp_variance_score
            avg_max_error += max_error
            avg_mae += mae
            avg_mse += mse
            avg_r2_score += r2_score(y_true, regressor.getPredictions())
            
            count += 1
            
        avg_explained_variance_score = avg_explained_variance_score / count
        avg_max_error = avg_max_error / count
        avg_mae = avg_mae / count
        avg_mse = avg_mse / count
        avg_r2_score = avg_r2_score / count
        
        results.append([data_to_test, float(avg_explained_variance_score), float(avg_max_error),
                   float(avg_mae), float(avg_mse), float(avg_r2_score)])
        
        
    
    print("ANN evaluation results")
    print("Average explained variance score:", avg_explained_variance_score)
    print("Average mean absolute error:", avg_mae)
    print("Average mean squared error:", avg_mse)
    print("Average r2 score:", avg_r2_score)
    
'''
def evaluateRandomForest():
    print("Evaluating Random Forest")
    for data in datasets:
        #Import the Dataset and separate X and y
        data_to_test = data + '.csv'
        X_before, y_before = importDataset(data_to_test)
        
        count = 0
        avg_explained_variance_score = 0
        avg_max_error = 0
        avg_mae = 0
        avg_mse = 0
        avg_r2_score = 0
       
        # prepare cross validation
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=3, shuffle=True)
        
        for train, test in kfold.split(X_before):
            print("Test:", count+1, " for", data_to_test)
            X_train, X_test = X_before.iloc[train], X_before.iloc[test]
            y_train, y_true = y_before[train], y_before[test]
            
            #feature scaling
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # run ANN
            from regressionAnalysis import sequentialNN
            regressor = sequentialNN(X_train, y_train, X_test, y_true)
            exp_variance_score, max_error, loss, mae, mse = regressor.getEvaluationMetrics()
            
            # get metrics
            avg_explained_variance_score += exp_variance_score
            avg_max_error += max_error
            avg_mae += mae
            avg_mse += mse
            avg_r2_score += r2_score(y_true, regressor.getPredictions())
            
            count += 1
            
        avg_explained_variance_score = avg_explained_variance_score / count
        avg_max_error = avg_max_error / count
        avg_mae = avg_mae / count
        avg_mse = avg_mse / count
        avg_r2_score = avg_r2_score / count
        
        results.append([data_to_test, float(avg_explained_variance_score), float(avg_max_error),
                   float(avg_mae), float(avg_mse), float(avg_r2_score)])
        
        
    
    print("ANN evaluation results")
    print("Average explained variance score:", avg_explained_variance_score)
    print("Average mean absolute error:", avg_mae)
    print("Average mean squared error:", avg_mse)
    print("Average r2 score:", avg_r2_score)
'''
    
def saveToFile():
    with open('evaluation.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(results)
         
evaluateANN() 
#evaluateRandomForest()
saveToFile()


