# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:04:53 2020

@author: Maria
"""
from algorithms import importDataset, splitAndScale, ANNregression, randomForest, svr
from sklearn.metrics import r2_score


def evaluateANN():
    
#    datasets = ['regAll', 'regBalanced', 'regEncoded',
#                'regEncodedBalanced', 'regNo365', 'regOnly365', 
#                'regSynthetic', 'regSyntheticWith365'
#               ]
    
#    for data in datasets:
    #Import the Dataset and separate X and y
#    data_to_test = data + '.csv'
    X_before, y_before = importDataset('regAll.csv')
    
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
        X_train, X_test = X_before.iloc[train], X_before.iloc[test]
        y_train, y_true = y_before[train], y_before[test]
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        from regressionAnalysis import sequentialNN
        regressor = sequentialNN(X_train, y_train, X_test, y_true)
        exp_variance_score, max_error, loss, mae, mse = regressor.getEvaluationMetrics()
        
        avg_explained_variance_score += exp_variance_score
        avg_max_error += max_error
        avg_mae += mae
        avg_mse += mse
        
        predictions = regressor.getPredictions()
        print(predictions)
        
        avg_r2_score += r2_score(y_true, predictions)
        
        count += 1
        
    avg_explained_variance_score = avg_explained_variance_score / count
    avg_max_error = avg_max_error / count
    avg_mae = avg_mae / count
    avg_mse = avg_mse / count
    avg_r2_score = avg_r2_score / count
    
    print("Regression metrics results")
    print("Average explained variance score:", avg_explained_variance_score)
    print("Average mean absolute error:", avg_mae)
    print("Average mean squared error:", avg_mse)
    print("Average r2 score:", avg_r2_score)
    
    
            
evaluateANN() 
        
