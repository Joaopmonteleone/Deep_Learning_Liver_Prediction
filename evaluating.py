# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:04:53 2020

@author: Maria
"""
from algorithms import importDataset, splitAndScale
from sklearn import metrics
from sklearn.metrics import r2_score, explained_variance_score, max_error, mean_absolute_error, mean_squared_error
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

kfold = KFold(n_splits=7, shuffle=True)
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
        data_to_test = "regression/" + data + '.csv'
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
        data_to_test = "regression/" + data + '.csv'
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
        data_to_test = "regression/" + data + '.csv'
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
claResults = [['', 'dataset','roc auc', 'accuracy', 'precision', 'recall', 'f1score']]

claDatasets = ['cla3monthSurvival', 'claSyntheticBalanced']

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
def encode(dataframe, columns):
   onehotencoder = OneHotEncoder(categorical_features = columns)
   encoded = onehotencoder.fit_transform(dataframe.values).toarray()
   return encoded

def encodeData(dataset):
    X_encoded = encode(dataset, [6, 7, 14, 21, 36]) 
    # etiology, portal thrombosis, pretransplant status performance, cause of death, cold ischemia time 
    y_encoded = dataset.iloc[:, 38]
    return X_encoded[:, :-1], y_encoded


def evaluateClassificationANN():
    claResults.append(["ANN"])
    for data in claDatasets:
        #Import the Dataset and separate X and y
        data_to_test = 'datasets/classification/' + data + '.csv'
        dataset = pd.read_csv(data_to_test)
        X_before, y_before = encodeData(dataset)
        # TRY WITH AND WITHOUT THIS
#        X_before = dataset.iloc[:, :-1].values
#        y_before = dataset.iloc[:, 38]
        
        count = 0
        avg_roc_auc = 0
        avg_accuracy = 0
        avg_precision = 0
        avg_recall = 0
        avg_f1score = 0
        
        fpr = 0
        tpr = 0
        threshold = 0
       
        for train, test in kfold.split(X_before):
            print("\nTest:", count+1, "for", data, "\n")
            X_train, X_test = X_before[train], X_before[test]
            y_train, y_true = y_before[train], y_before[test]
            
            #feature scaling
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # run ANN
            from ANN import ANN
            classifier = ANN(X_train,y_train,10,500,1) # batch_size, epochs, output layer hidden units

            y_prob, y_pred, accuracy = classifier.predict_all(X_test, y_true)
            
            fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
            roc_auc = metrics.auc(fpr, tpr)
            
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
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % avg_roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    name = data + 'roc.png'
    plt.savefig(name)
    
    
def evaluateSVM():
    claResults.append(["SVM"])
    for data in claDatasets:
        #Import the Dataset and separate X and y
        data_to_test = 'datasets/classification/' + data + '.csv'
        dataset = pd.read_csv(data_to_test)
        X_before, y_before = encodeData(dataset)
#        X_before = dataset.iloc[:, :-1].values
#        y_before = dataset.iloc[:, 38]
        
        count = 0
        avg_roc_auc = 0
        avg_accuracy = 0
        avg_precision = 0
        avg_recall = 0
        avg_f1score = 0
        
        fpr = 0
        tpr = 0
        threshold = 0
       
        for train, test in kfold.split(X_before):
            print("Test:", count+1, " for", data)
            X_train, X_test = X_before[train], X_before[test]
            y_train, y_true = y_before[train], y_before[test]
            
            #feature scaling
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # run SVM
            from svm import svm
            svm = svm(X_train, y_train, X_test, y_true)
            y_pred = svm.getPredictions()
            
            fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
            roc_auc = metrics.auc(fpr, tpr)
            
            # get metrics
            avg_roc_auc += roc_auc
            avg_accuracy += svm.getAccuracy()
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
        
    print("\nSVM evaluation results")
    print("Average ROC AUC:", avg_roc_auc)
    print("Average accuracy:", avg_accuracy)
    print("Average precision:", avg_precision)
    print("Average recall:", avg_recall)
    print("Average f1 score:", avg_f1score)
    
    
    
def saveToFileCla():
    with open('ClaEvaluation.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(claResults)
    
#evaluateClassificationANN()
#evaluateSVM()
#saveToFileCla()



###############################################
#  Evaluation with previous d-r algorithms    #
###############################################
recipientDatasets = ['rec1', 'rec2', 'rec3', 'rec4', 'rec5']
predict_results = [['','ANN','RF', 'SVR']]

def findBestMatch():
    print("\nEvaluating different recipients")
    
    X_before, y_before = importDataset('regression/regSyntheticWith365.csv')
    X_train, X_test, y_train, y_true = splitAndScale(X_before, y_before)
    
    # Train models with synthetic dataset
    from regressionAnalysis import sequentialNN
#    sequentialNN(X_train, y_train, X_test, y_true)
    ann = tf.keras.models.load_model('models/ann.h5')
    from randomForest import randomForest
#    randomForest(X_train, y_train, X_test, y_true, X_before)
    rf = joblib.load('models/rf.sav')
    from svr import svr
#    svr(X_train, y_train, X_test, y_true)
    svr = joblib.load('models/svr.sav')
    
    MLmodels = [ann, rf, svr]
    
    for data in recipientDatasets:
        predict_results.append([data])
        print("Predicting for",data)
        dataset = pd.read_csv('datasets/' + data + '.csv')
        to_predict = dataset.iloc[:, :-1].values # get all columns except last one (actual value)
  
        for row in to_predict:
            transform = scaler.fit_transform(row.reshape(-1, 1))
            prediction = ['']
            for model in MLmodels:
                new_pred = model.predict(transform.reshape(1, -1))
                if 'Sequential' in str(type(model)):
                    prediction.append(new_pred[0][0])
                else:
                    prediction.append(new_pred[0])
                
        predict_results.append(prediction)
    print('Predictions saved to file RecipientsPredictions.csv')
                
def saveToFilePredictions():
    with open('datasets/evaluation/RecipientsPredictions.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(predict_results)
    
findBestMatch()
saveToFilePredictions()










