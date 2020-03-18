# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:04:53 2020

@author: Maria
"""
from algorithms import importDataset
from sklearn import metrics
from sklearn.metrics import r2_score, explained_variance_score, max_error, mean_absolute_error, mean_squared_error
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
#import matplotlib.pyplot as plt

kfold = KFold(n_splits=3, shuffle=True)
scaler = MinMaxScaler()


###############################################
#             Regression Models               #
###############################################
regResults = [['', 'dataset', 'variance score', 'max error', 'mae', 'mse', 'r2 score']]

regDatasets = ['regAll', 'regBalanced', 'regEncoded',
            'regEncodedBalanced', 'regNo365', 'regOnly365', 
            'regSynthetic', 'regSyntheticWith365'
           ]

def evaluateANN():
    regResults.append(["Results for ANN"])
    for data in regDatasets:
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
        
        regResults.append(['', data_to_test, float(avg_explained_variance_score), float(avg_max_error),
                   float(avg_mae), float(avg_mse), float(avg_r2_score)])
        
        
    
    print("ANN evaluation results")
    print("Average explained variance score:", avg_explained_variance_score)
    print("Average mean absolute error:", avg_mae)
    print("Average mean squared error:", avg_mse)
    print("Average r2 score:", avg_r2_score)
    



def evaluateRandomForest():
    print("\nEvaluating Random Forest")
    regResults.append(["Results for Random Forest"])
    for data in regDatasets:
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
        
        regResults.append(['', data_to_test, float(avg_explained_variance_score), float(avg_max_error),
                   float(avg_mae), float(avg_mse), float(avg_r2_score)])
        
        
    
    print("Random Forest evaluation results")
    print("Average explained variance score:", avg_explained_variance_score)
    print("Average mean absolute error:", avg_mae)
    print("Average mean squared error:", avg_mse)
    print("Average r2 score:", avg_r2_score)



def evaluateSVR():
    print("\nEvaluating SVR")
    regResults.append(["Results for SVR"])
    for data in regDatasets:
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
        
        regResults.append(['', data_to_test, float(avg_explained_variance_score), float(avg_max_error),
                   float(avg_mae), float(avg_mse), float(avg_r2_score)])
        
        
    
    print("Random Forest evaluation results")
    print("Average explained variance score:", avg_explained_variance_score)
    print("Average mean absolute error:", avg_mae)
    print("Average mean squared error:", avg_mse)
    print("Average r2 score:", avg_r2_score)
    
    
def saveToFile():
    with open('RegEvaluation.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(regResults)
        
def runRegressionEvaluation():
    evaluateANN() 
    evaluateRandomForest()
    evaluateSVR()
    saveToFile()
    

#runRegressionEvaluation()


###############################################
#           Classification Models             #
###############################################
claResults = [['', 'dataset', 'roc', 'auc roc', 'accuracy', 'confusion matrix']]

claDatasets = ['claAll', 'claBalanced', 'claSyntheticWith4', 'claNo4', 'claOnly4', 'claSynthetic'
               ]

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
def encode(dataframe, columns):
   onehotencoder = OneHotEncoder(categorical_features = columns)
   encoded = onehotencoder.fit_transform(dataframe.values).toarray()
   return encoded

def encodeData(dataset):
    X_encoded = encode(dataset, [6, 7, 14, 21, 36]) 
    # etiology, portal thrombosis, pretransplant status performance, cause of death, cold ischemia time 
    y_encoded = encode(dataset, [38])
    y_encoded = y_encoded[:, 0:4]
    return X_encoded[:, :-1], y_encoded

def evaluateClassificationANN():
    claResults.append(["Results for ANN"])
    for data in claDatasets:
        #Import the Dataset and separate X and y
        data_to_test = 'datasets/' + data + '.csv'
        dataset = pd.read_csv(data_to_test)
        X_before, y_before = encodeData(dataset)
        
        print("\n\n\ny_before[0, 3]", y_before[0, 3])
        
        if y_before[0,3] == 1 or y_before[0,3] == 0:
            y_4 = y_before[:, 3]
            print("\n\nTesting for 1 year survival\n\n")
        else:
            y_4 = y_before[:, 1]
            print(y_before)
            print("\n\nTesting for 3 months survival\n\n")
        
        count = 0
        avg_roc_auc = 0
        avg_accuracy = 0
        avg_precision = 0
        avg_recall = 0
        avg_f1score = 0
       
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
                             10, 1, 1) # batch_size, epochs, output layer hidden units

            y_prob, y_pred, accuracy = classifier.predict_all(X_test, y_true)
            
            print(metrics.roc_curve(y_true, y_pred))
            fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
            roc_auc = metrics.auc(fpr, tpr)
            
#            plt.title('Receiver Operating Characteristic')
#            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#            plt.legend(loc = 'lower right')
#            plt.plot([0, 1], [0, 1],'r--')
#            plt.xlim([0, 1])
#            plt.ylim([0, 1])
#            plt.ylabel('True Positive Rate')
#            plt.xlabel('False Positive Rate')
#            plt.show()
            
            # get metrics
            avg_roc_auc += roc_auc
            avg_accuracy += metrics.accuracy_score(y_true, y_pred)
            avg_precision += metrics.precision_score(y_true, y_pred)
            avg_recall += metrics.recall_score(y_true, y_pred)
            avg_f1score += metrics.f1_score(y_true, y_pred)
            
            count += 1

        avg_roc_auc = avg_roc_auc / count
        avg_accuracy = avg_accuracy / count
        avg_precision = avg_precision / count
        avg_recall = avg_recall / count
        avg_f1score = avg_f1score / count
        
        claResults.append(['', data_to_test, float(avg_roc_auc), float(avg_accuracy),
                        float(avg_precision), float(avg_recall), float(avg_f1score)
                        ])
        
        
    print("\nANN evaluation results")
    print("Average ROC AUC:", avg_roc_auc)
    print("Average accuracy:", avg_accuracy)
    print("Average precision:", avg_precision)
    print("Average recall:", avg_recall)
    print("Average f1 score:", avg_f1score)
    
def saveToFileCla():
    with open('ClaEvaluation.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(claResults)
    
evaluateClassificationANN()
saveToFileCla()

