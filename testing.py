
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold
import sklearn.svm as svm

from tensorflow import keras
from liver_prediction_ANN import liver_prediction_ANN

import csv
import pandas as pd
import numpy as np

from Utility import Metric, generateMetric, generateMeanPredictions, showMetrics



def runNeuralNetworkExperiment(trainingDataset, testingDataset, nTimes, learningRate = 0.001, numDenseLayers = 1, numDenseNodes = 10, activation = 'relu', numFeatures = 55, epochs = 100, optimizer = 'adam'):

    df = pd.read_csv(trainingDataset, header=0)
    # attributes that are features to predict PV output
    # x = df[[ 'month', 'day', 'hour', 'IDN', 'I', 'oktas', 'visibility', 'hsd', 'temp']]
    df = df[['month','day','hour','IDN','I','oktas','visibility','hsd', 'temp','pv']]

    x = df[['I', 'oktas', 'visibility', 'hsd','temp']].to_numpy()
    y = df[['pv']].to_numpy()

    numFeatures = x.shape[1]
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    j = 0
    avgTestMAE = 0
    avgTestMSE = 0
    avgTestRMSE = 0
    avgTestVarianceScore = 0
    metricsList = []

    for n in range(nTimes):

        print("Test Number: " + str(n))
        ANN = NeuralNetwork(learningRate, numDenseLayers, numDenseNodes, activation, numFeatures, epochs, optimizer)

        i = 0
        v = 0
        mae = 0
        mse = 0
        rmse = 0

        kf = RepeatedKFold(n_splits=5, n_repeats=1)

        for trainIndex, testIndex in kf.split(x):

            print("TRAIN:", trainIndex, "TEST:", testIndex)

            xTrain, xTest = x[trainIndex], x[testIndex]
            yTrain, yTest = y[trainIndex], y[testIndex]

            scaler.fit(xTrain)

            xTrain = scaler.transform(xTrain)
            xTest = scaler.transform(xTest) 
    
            earlyStop = keras.callbacks.EarlyStopping(monitor = 'loss', patience = 5)
            ANN.fit(xTrain, yTrain, epochs = epochs, verbose = 0, callbacks = [earlyStop])

            predictions = ANN.predict(xTest).flatten()   

            v = v + metrics.r2_score(yTest, predictions)
            mae = mae + metrics.mean_absolute_error(yTest, predictions)
            mse = mse + metrics.mean_squared_error(yTest, predictions)
            rmse = rmse + np.sqrt(metrics.mean_squared_error(yTest, predictions))

            i = i + 1

        v = v / i
        mae = mae / i
        rmse = rmse / i
        mse = mse / i
        
        print('VALIDATION RESULTS')
        print("Average Mean Absolute Error:", mae)
        print("Average Mean Squared Error:", mse)
        print("Average Root Mean Squared Error:", rmse)
        print("Average Variance Score:", v)

        testDataset = pd.read_csv(testingDataset, header=0)

        df = testDataset[['month','day','hour','IDN','I','oktas', 'hsd','temp','visibility','pv']]   

        testX = df[['I', 'oktas', 'visibility', 'temp', 'hsd']].to_numpy()
        testY = df[['pv']].to_numpy()

        scaler.fit(testX)
        testX = scaler.transform(testX)
        testPred = ANN.predict(testX).flatten()

        tempTestMAE = metrics.mean_absolute_error(testY, testPred)
        tempTestMSE = metrics.mean_squared_error(testY, testPred)
        tempTestRMSE = np.sqrt(metrics.mean_squared_error(testY, testPred))
        tempTestVarianceScore = metrics.r2_score(testY, testPred)

        avgTestMAE = avgTestMAE + metrics.mean_absolute_error(testY, testPred)
        avgTestMSE = avgTestMSE + metrics.mean_squared_error(testY, testPred)
        avgTestRMSE = avgTestRMSE + np.sqrt(metrics.mean_squared_error(testY, testPred))
        avgTestVarianceScore = avgTestVarianceScore + metrics.r2_score(testY, testPred)

        metric = Metric(tempTestMAE, tempTestMSE, tempTestRMSE, tempTestVarianceScore)

        metricsList.append(metric)
        showMetrics(testY, testPred, message = "TEST RESULTS")

        predictions = testPred.tolist()
        yTestList = testY.tolist()
        j = j + 1

        generateTestPredictions("TestPredictionsKFoldNeuralNetwork" + str(n) + ".csv", predictions, yTestList)      

        if n != nTimes - 1:
            del ANN
        elif n ==  nTimes - 1:
            ANN.exportModelInfo()
            del ANN
        keras.backend.clear_session()
    
    avgTestMAE = avgTestMAE / j
    avgTestMSE = avgTestMSE / j
    avgTestRMSE = avgTestRMSE / j
    avgTestVarianceScore = avgTestVarianceScore / j

    testMetric = Metric(avgTestMAE, avgTestMSE, avgTestRMSE, avgTestVarianceScore)

    generateMetric([testMetric], "FinalTestResultsKFoldNeuralNetwork")
    generateMetric(metricsList, "AllTestResultsKFoldNeuralNetwork")
    generateMeanPredictions("TestPredictionsKFoldNeuralNetwork", nTimes)

    print("Average Test Results:")
    print("Average Mean Absolute Error:", avgTestMAE)
    print("Average Mean Squared Error:", avgTestMSE)
    print("Average Root Mean Squared Error:", avgTestRMSE)
    print("Average Variance Score:", avgTestVarianceScore)


def runSupportVectorMachineExperiment(trainingDataset, testingDataset, nTimes):
    
    df = pd.read_csv(trainingDataset, header=0)

    # attributes that are features to predict PV output
    # x = df[[ 'month', 'day', 'hour', 'IDN', 'I', 'oktas', 'visibility', 'hsd', 'temp']]

    df = df[['month','day','hour','I', 'oktas', 'visibility', 'hsd', 'temp', 'pv']]

    x = df[['I', 'oktas', 'visibility', 'hsd', 'temp']].to_numpy()
    y = df[['pv']].to_numpy()

    num_features = x.shape[1]
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))


    j = 0
    avgTestMAE = 0
    avgTestMSE = 0
    avgTestRMSE = 0
    avgTestVarianceScore = 0
    metricsList = []
    predictionList = []

    for n in range(nTimes):

        print('Test Number: ' + str(n))
        svr = svm.SVR(C = 250, gamma = 'auto', cache_size = 500, kernel = 'rbf', tol = 1e-3)

        print("Support Vector Properties: \n" + str(svr))

        i = 0
        v = 0
        mae = 0
        mse = 0
        rmse = 0

        kf = RepeatedKFold(n_splits=5, n_repeats=1)

        for trainIndex, testIndex in kf.split(x):

            print("TRAIN:", trainIndex, "TEST:", testIndex)

            xTrain, xTest = x[trainIndex], x[testIndex]
            yTrain, yTest = y[trainIndex], y[testIndex]

            scaler.fit(xTrain)

            xTrain = scaler.transform(xTrain)
            xTest = scaler.transform(xTest) 

            svr.fit(xTrain, yTrain.ravel())

            yPred = svr.predict(xTest)

            showMetrics(yTest, yPred, message= "Fold Results")

            v = v + metrics.r2_score(yTest, yPred)
            mae = mae + metrics.mean_absolute_error(yTest, yPred)
            mse = mse + metrics.mean_squared_error(yTest, yPred)
            rmse = rmse + np.sqrt(metrics.mean_squared_error(yTest, yPred))

            i = i + 1


        v = v / i
        mae = mae / i
        rmse = rmse / i
        mse = mse / i

        print("VALIDATION RESULTS")
        print("Average Mean Absolute Error:", mae)
        print("Average Mean Squared Error:", mse)
        print("Average Root Mean Squared Error:", rmse)
        print("Average Variance Score:", v)

        testDataset = pd.read_csv(testingDataset, header=0)

        df = testDataset[['month','hour','I', 'oktas', 'visibility', 'hsd', 'temp', 'pv']]

        testX = df[['I', 'oktas', 'visibility', 'hsd', 'temp']].to_numpy()
        testY = df[['pv']].to_numpy()

        scaler.fit(testX)
        testX = scaler.transform(testX)

        testPred = svr.predict(testX)

        print("TEST RESULTS:")       

        tempTestMAE = metrics.mean_absolute_error(testY, testPred)
        tempTestMSE = metrics.mean_squared_error(testY, testPred)
        tempTestRMSE = np.sqrt(metrics.mean_squared_error(testY, testPred))
        tempTestVarianceScore = metrics.r2_score(testY, testPred)

        avgTestMAE = avgTestMAE + metrics.mean_absolute_error(testY, testPred)
        avgTestMSE = avgTestMSE + metrics.mean_squared_error(testY, testPred)
        avgTestRMSE = avgTestRMSE + np.sqrt(metrics.mean_squared_error(testY, testPred))
        avgTestVarianceScore = avgTestVarianceScore + metrics.r2_score(testY, testPred)
 
        metric = Metric(tempTestMAE, tempTestMSE, tempTestRMSE, tempTestVarianceScore)
        metricsList.append(metric)
        showMetrics(testY, testPred, message = "TEST RESULTS")

        predictions = testPred.tolist()
        yTestList = testY.tolist() 
        j = j + 1

        generateTestPredictions("TestPredictionsKFoldSVR" + str(n) + ".csv", predictions, yTestList) 

        del svr 
    
    avgTestMAE = avgTestMAE / j
    avgTestMSE = avgTestMSE / j
    avgTestRMSE = avgTestRMSE / j
    avgTestVarianceScore = avgTestVarianceScore / j

    testMetric = Metric(avgTestMAE, avgTestMSE, avgTestRMSE, avgTestVarianceScore)

    generateMetric([testMetric], "FinalTestResultsKFoldSVR")
    generateMetric(metricsList, 'AllTestResultsKFoldSVR')
    generateMeanPredictions("TestPredictionsKFoldSVR", nTimes)

    print("Average Test Results:")
    print("Average Mean Absolute Error:", avgTestMAE)
    print("Average Mean Squared Error:", avgTestMSE)
    print("Average Root Mean Squared Error:", avgTestRMSE)
    print("Average Variance Score:", avgTestVarianceScore)
