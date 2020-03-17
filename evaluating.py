# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:04:53 2020

@author: Maria
"""
from algorithms import importDataset, encodeData
from sklearn import metrics
from sklearn.metrics import r2_score, explained_variance_score, max_error, mean_absolute_error, mean_squared_error
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

kfold = KFold(n_splits=3, shuffle=True)
scaler = MinMaxScaler()


###############################################
#             Regression Models               #
###############################################
results = [['', 'dataset', 'variance score', 'max error', 'mae', 'mse', 'r2 score']]

datasets = ['regAll', 'regBalanced', 'regEncoded',
            'regEncodedBalanced', 'regNo365', 'regOnly365', 
            'regSynthetic', 'regSyntheticWith365'
           ]

def evaluateANN():
    results.append(["Results for ANN"])
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
       
        for train, test in kfold.split(X_before):
            print("Test:", count+1, " for", data_to_test)
            X_train, X_test = X_before.iloc[train], X_before.iloc[test]
            y_train, y_true = y_before[train], y_before[test]
            
            #feature scaling
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
        
        results.append(['', data_to_test, float(avg_explained_variance_score), float(avg_max_error),
                   float(avg_mae), float(avg_mse), float(avg_r2_score)])
        
        
    
    print("ANN evaluation results")
    print("Average explained variance score:", avg_explained_variance_score)
    print("Average mean absolute error:", avg_mae)
    print("Average mean squared error:", avg_mse)
    print("Average r2 score:", avg_r2_score)
    



def evaluateRandomForest():
    print("\nEvaluating Random Forest")
    results.append(["Results for Random Forest"])
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
       
        for train, test in kfold.split(X_before):
            print("Test:", count+1, "for", data_to_test)
            X_train, X_test = X_before.iloc[train], X_before.iloc[test]
            y_train, y_true = y_before[train], y_before[test]
            
            #feature scaling
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # run algorithm
            from randomForest import randomForest
            rfModel = randomForest(X_train, y_train, X_test, y_true, X_before)
            predictions = rfModel.getPredictions()
            
            # get metrics
            avg_explained_variance_score += explained_variance_score(y_true, predictions)
            avg_max_error += max_error(y_true, predictions)
            avg_mae += mean_absolute_error(y_true, predictions)
            avg_mse += mean_squared_error(y_true, predictions)
            avg_r2_score += r2_score(y_true, predictions)
            
            count += 1
            
        avg_explained_variance_score = avg_explained_variance_score / count
        avg_max_error = avg_max_error / count
        avg_mae = avg_mae / count
        avg_mse = avg_mse / count
        avg_r2_score = avg_r2_score / count
        
        results.append(['', data_to_test, float(avg_explained_variance_score), float(avg_max_error),
                   float(avg_mae), float(avg_mse), float(avg_r2_score)])
        
        
    
    print("Random Forest evaluation results")
    print("Average explained variance score:", avg_explained_variance_score)
    print("Average mean absolute error:", avg_mae)
    print("Average mean squared error:", avg_mse)
    print("Average r2 score:", avg_r2_score)



def evaluateSVR():
    print("\nEvaluating SVR")
    results.append(["Results for SVR"])
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
        
        for train, test in kfold.split(X_before):
            print("Test:", count+1, "for", data_to_test)
            X_train, X_test = X_before.iloc[train], X_before.iloc[test]
            y_train, y_true = y_before[train], y_before[test]
            
            #feature scaling
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # run algorithm
            from svr import svr
            svr = svr(X_train, y_train, X_test, y_true)
            predictions = svr.getPredictions()
            
            # get metrics
            avg_explained_variance_score += explained_variance_score(y_true, predictions)
            avg_max_error += max_error(y_true, predictions)
            avg_mae += mean_absolute_error(y_true, predictions)
            avg_mse += mean_squared_error(y_true, predictions)
            avg_r2_score += r2_score(y_true, predictions)
            
            count += 1
            
        avg_explained_variance_score = avg_explained_variance_score / count
        avg_max_error = avg_max_error / count
        avg_mae = avg_mae / count
        avg_mse = avg_mse / count
        avg_r2_score = avg_r2_score / count
        
        results.append(['', data_to_test, float(avg_explained_variance_score), float(avg_max_error),
                   float(avg_mae), float(avg_mse), float(avg_r2_score)])
        
        
    
    print("Random Forest evaluation results")
    print("Average explained variance score:", avg_explained_variance_score)
    print("Average mean absolute error:", avg_mae)
    print("Average mean squared error:", avg_mse)
    print("Average r2 score:", avg_r2_score)
    
    
def saveToFile():
    with open('RegEvaluation.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(results)
        
def runRegressionEvaluation():
    evaluateANN() 
    evaluateRandomForest()
    evaluateSVR()
    saveToFile()
    

#runRegressionEvaluation()


###############################################
#           Classification Models             #
###############################################
results = [['', 'dataset', 'roc', 'auc roc', 'accuracy', 'confusion matrix']]

datasets = ['claAll', #'claBalanced', 'claNo4', 'claOnly4', 
#            'claSynthetic', 'claSyntheticWith4'
           ]

def evaluateClassificationANN():
    results.append(["Results for ANN"])
    for data in datasets:
        #Import the Dataset and separate X and y
        data_to_test = 'datasets/' + data + '.csv'
        dataset = pd.read_csv(data_to_test)
        X_before, y_before = encodeData(dataset)
        
        y_1 = y_before[:, 0]
        y_2 = y_before[:, 1]
        y_3 = y_before[:, 2]
        y_4 = y_before[:, 3]
        
        count = 0
        avg_roc = 0
#        avg_roc_auc = 0
#        avg_accuracy = 0
#        avg_cm = 0
       
        for train, test in kfold.split(X_before):
            print("Test:", count+1, " for", data_to_test)
            X_train, X_test = X_before[train], X_before[test]
            y_train, y_true = y_4[train], y_4[test]
            
            # change y to test different outputs?
                # predicting 1-2-3-4
                # predicting 0-0-0-1
                # predicting just one binary column
            
            #feature scaling
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # run ANN
            from ANN import ANN
            activation_output = ['softmax', 'sigmoid'] # softmax for 4, sigmoid for binary
            loss = ['categorical_crossentropy', 'binary_crossentropy'] 
            classifier = ANN(X_train,y_train,'relu',activation_output[1],'rmsprop',loss[1],
                             10, 5, 1) # batch_size, epochs, output layer hidden units

            y_prob, y_pred, accuracy = classifier.predict_all(X_test, y_true)
            
            print(int(y_true[0]))
            print(y_pred[0])
            print(metrics.roc_curve(y_true, y_pred))
            
#            metrics.plot_roc_curve(classifier, X_test, y_true)
#            plt.show()
            
            fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
            roc_auc = metrics.auc(fpr, tpr)
            
            
            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()
            


            # get metrics
            avg_roc += roc_auc
#            avg_max_error += max_error
#            avg_mae += mae
#            avg_mse += mse
#            avg_r2_score += r2_score(y_true, regressor.getPredictions())
            
            count += 1

        avg_roc = avg_roc / count
#        avg_max_error = avg_max_error / count
#        avg_mae = avg_mae / count
#        avg_mse = avg_mse / count
#        avg_r2_score = avg_r2_score / count
        
        results.append(['', data_to_test, float(avg_roc)#, float(avg_max_error),
                 #  float(avg_mae), float(avg_mse), float(avg_r2_score)
                       ])
        
        
    
    print("ANN evaluation results")
    print("Average ROC:", avg_roc)
#    print("Average mean absolute error:", avg_mae)
#    print("Average mean squared error:", avg_mse)
#    print("Average r2 score:", avg_r2_score)
    

evaluateClassificationANN()

