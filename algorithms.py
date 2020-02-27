# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:32:59 2020

@author: Maria
"""
###############################################
#             Data Preprocessing              #
###############################################

# Importing the dataset
import pandas as pd
dataset = pd.read_csv('datasets/regSynthetic.csv')

X_before = dataset.iloc[:, :-1] # all rows, all columns except last result and 3 months answer - (1198, 39)
y_before = dataset.iloc[:, (dataset.values.shape[1]-1)].values # all rows, last column (result) keep a record to compare later

'''BELOW IS ONLY USED FOR CLASSIFICATION DATA'''
# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
def encode(dataframe, columns):
   onehotencoder = OneHotEncoder(categorical_features = columns)
   encoded = onehotencoder.fit_transform(dataframe.values).toarray()
   return encoded
X_encoded = encode(X_before, [6, 7, 14, 21, 36]) # etiology, portal thrombosis, pretransplant status performance, cause of death, cold ischemia time 

# encoding the output FOR CLASSIFICATION
y_encoded = encode(dataset, [38])
y_encoded = y_encoded[:, 0:4]

# Separating each column to predict separate classes
y_1 = y_encoded[:, 0]
y_2 = y_encoded[:, 1]
y_3 = y_encoded[:, 2]
y_4 = y_encoded[:, 3]


selectedY = y_before
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_before, 
                                                    selectedY, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)


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
                 optimizer[1], loss[1], 
                 10, 500, 1) # batch_size, epochs, output layer hidden units

y_pred, y_bool, accuracy = classifier.predict_all(X_test, y_test)

# Evaluate ANN - only with binary prediction
accuracies, mean, variance = ANN.evaluate(X_train, y_train)

# Grid Search
best_parameters, best_accuracy = ANN.grid_search(X_train, y_train) 




###############################################
#         Regression Random Forest            #
############################################### 
from randomForest import randomForest
# RandomForestRegressor
rfModel = randomForest(X_train, y_train, X_test, y_test, X_before)
# Get top 15 instances
importances = rfModel.getImportance()
# Plot graph
randomForest.plotRandomForest(y_test, rfModel.predictions)
randomForest.makeTree(rfModel)
    

###############################################
#               ANN Regression                #
###############################################
from regressionAnalysis import sequentialNN
regressor = sequentialNN(X_train, y_train, X_test, y_test)
regressor.visualizeMSEoverEPOCHS()
regressor.visualizePredictionsVsActual()


###############################################
#          Support Vector Regression          #
###############################################
from svr import svr
svr = svr(X_train, y_train, X_test, y_test)
predictions = svr.getPredictions()
svr.svr_graph()
mape = svr.getMAPE()
best_params = svr.grid_search()


###############################################
#           Support Vector Machine            #
###############################################
''' claBalanced - y_before  '''
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
#                    READ                     #
###############################################

#https://scikit-learn.org/stable/modules/multiclass.html
#https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
#https://towardsdatascience.com/synthetic-data-generation-a-must-have-skill-for-new-data-scientists-915896c0c1ae












