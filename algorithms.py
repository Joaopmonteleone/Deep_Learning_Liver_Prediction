# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:32:59 2020

@author: Maria
"""
###############################################
#             Data Preprocessing              #
###############################################

dataset = []


# Importing the dataset
import pandas as pd
def importDataset(dataset):
    X_before = []
    y_before = []
    location = 'datasets/'+dataset
    dataset = pd.read_csv(location)
    X_before = dataset.iloc[:, :-1] # all rows, all columns except last result and 3 months answer - (1198, 39)
    y_before = dataset.iloc[:, (dataset.values.shape[1]-1)].values # all rows, last column (result) keep a record to compare later
    return X_before, y_before


# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
def encode(dataframe, columns):
   onehotencoder = OneHotEncoder(categorical_features = columns)
   encoded = onehotencoder.fit_transform(dataframe.values).toarray()
   return encoded

def encodeData(dataset, X_before):
    X_encoded = encode(X_before, [6, 7, 14, 21, 36]) # etiology, portal thrombosis, pretransplant status performance, cause of death, cold ischemia time 
    # encoding the output FOR CLASSIFICATION
    y_encoded = encode(dataset, [38])
    y_encoded = y_encoded[:, 0:4]
    return X_encoded, y_encoded

def splitAndScale(X_before, y_before):
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_before, 
                                                        y_before, 
                                                        test_size = 0.2, 
                                                        random_state = 0)
    
    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


###############################################
#               Regression                    #
###############################################
def ANNregression(X_train, y_train, X_test, y_test):
    print("Training ANN on dataset...")
    from regressionAnalysis import sequentialNN
    regressor = sequentialNN(X_train, y_train, X_test, y_test)
    #regressor.visualizeMSEoverEPOCHS()
    regressor.visualizePredictionsVsActual()
    exp_variance_score, max_error, loss, mae, mse = regressor.getEvaluationMetrics()
    print("\nMean absolute error of predictions:", int(mae), "days")
    # Do Grid Search
#    best_params, best_score = gridSearch(X_train, y_train)
    return mae

def randomForest(X_train, y_train, X_test, y_test, X_before):
    from randomForest import randomForest
    # RandomForestRegressor
    rfModel = randomForest(X_train, y_train, X_test, y_test, X_before)
    randomForest.plotRandomForest(y_test, rfModel.predictions)
    print("\nMean absolute error of predictions:", int(rfModel.getMAE()), "days")
    # Get top 15 instances
    print("\n-- Variable Importances --")
    importances = rfModel.getImportance()
    # Plot graph
    
#    randomForest.makeTree(rfModel)
    # Grid Search
#    rfModel.gridSearch()
    return rfModel.getMAE(), importances
    

def svr(X_train, y_train, X_test, y_test):
    from svr import svr
    svr = svr(X_train, y_train, X_test, y_test)
    svr.svr_graph()
    print("\nMean absolute error of predictions:", int(svr.getMAE()),"days")
#    best_params = svr.grid_search()
    return svr.getMAE()





'''
###############################################
#           Support Vector Machine            #
###############################################
# claBalanced - y_before  
from svm import svm
svm = svm(X_train, y_train, X_test, y_test)
#predictions = svm.getPredictions()
accuracy = svm.getAccuracy()
#class_report = svm.getClassificationReport()
#cm = svm.getMultilabelCM()
svm.svm_graph()

scores = {}
for i in range(30):
    params, estimator, score = svm.grid_search()
    scores[i] = score
maxval = max(scores.values())
res = [(k, v) for k, v in scores.items() if v == maxval]
print("Highest score:", res)


###############################################
#          ANN for classification             #
###############################################
from ANN import ANN

activation_output = ['softmax', 'sigmoid'] #softmax for 4, sigmoid for binary
optimizer = ['adagrad', 'adam', 'rmsprop', 'sgd'] # adagrad, adam, rmsprop, sgd
loss = ['categorical_crossentropy', 'binary_crossentropy', # binary or categorical
        'sparse_categorical_crossentropy']# use 5 output units

classifier = ANN(X_train, y_train, 
                 'relu', activation_output[1], 
                 optimizer[3], loss[1], 
                 10, 500, 1) # batch_size, epochs, output layer hidden units

cvscores = classifier.cross_validate(X_train, y_train, X_test, y_test)


y_pred, y_bool, accuracy = classifier.predict_all(X_test, y_test)

# Evaluate ANN - only with binary prediction
accuracies, mean, variance = ANN.evaluate_model(X_train, y_train)

# Grid Search
best_parameters, best_accuracy = ANN.gridSearch(X_train, y_train) 

'''








