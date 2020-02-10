# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:13:53 2020

@author: Maria
"""


# For the ANN building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
# For making predictions
from sklearn.metrics import confusion_matrix
# For evaluating
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score
# For tuning
from sklearn.model_selection import GridSearchCV


class ANN:
   def __init__(self, X_train, y_train, activation_hidden, activation_output, optimizer, loss, batch_size, epochs, output_units):

      classifier = Sequential()
      classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = activation_hidden, input_dim = X_train.shape[1]))
      classifier.add(Dropout(rate=0.1)) #EXPERIMENT WITH AND WITHOUT THIS
      classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = activation_hidden)) #relu
      classifier.add(Dropout(rate=0.1))
      classifier.add(Dense(units = output_units, kernel_initializer = 'uniform', activation = activation_output))
       
      classifier.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
  
      classifier.fit(X_train, y_train, batch_size = batch_size, epochs = epochs) 
       
      self.model = classifier
 
   ###############################################
   #             Evaluating the ANN              #
   ###############################################
   def evaluateANN(self, X_train, y_train, 
                   activation_hidden, activation_output, 
                   optimizer, loss, batch_size, epochs):
      
       classifier = Sequential()
       classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = activation_hidden, input_dim = X_train.shape[1]))
       classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = activation_hidden))
       classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = activation_output))
       classifier.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
       
       classifier = KerasClassifier(build_fn = classifier, batch_size = batch_size, epochs = epochs)
   
       ''' Same as the ANN architecture but instead of fitting to X_Train and y_train, it is vuilt
       on 10 different training folds, each time measuring the model performance on one test fold. ''' 
   
       # this line takes a while
       accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
       ''' estimator: the object to use to fit the data (classifier)
       X = the data to fit (X_train)
       y = to train a model, you need y's to understand correlations
       cv: number of folds in k-fold cross validation, 10 is recommended
       n_jobs: number of CPUs to use to do the computations, -1 means 'all CPUs' to run parallel
           computations and run the training faster'''
   
       mean = accuracies.mean() # find the average of the accuracies
       variance = accuracies.std() # find the variance of the accuracies (if < 1% = rather low variance)
       
       return classifier, accuracies, mean, variance

    # WITH BALANCED DATASET CATEGORICAL
    # mean 0.3441 - very very very low
    # variance 0.08 is < 1% so good 
    
    # WITH BALANCED DATASET BINARY - 4
    # mean 0.77%
    # variance 0.07%

   def predict_one(self, X_test):
       y_pred = self.model.predict(X_test)
       return y_pred
    
   def predict_all(self, X_test):
      y_pred = self.predict_one(X_test)
      y_bool = []   
      cm = 0
      if self.y_test.shape[1] == 1: # binary prediction (1s or 0s) ONLY WORKS WHEN PREDICTING ONE, NOT 4
         for n in y_pred:
            if n > 0.75:
               n = 1
            else:
               n = 0
            y_bool.append(n)
      # Making the Confusion Matrix - not valid for categorical outputs, only for binary
      cm = confusion_matrix(self.y_test, y_bool)
      return y_pred, y_bool, cm



###############################################
#        Improving & Tuning the ANN           #
###############################################
''' PARAMETER TUNING - THE GRID SEARCH TECHNIQUE
    When tuning the optimizer, the parameters to study must be passed through the function 
    Classification metrics can't handle a mix of multilabel-indicator and multiclass targets
    ONLY USE WHEN PREDICTING BINARY OUTPUTS '''


def build_classifier(optimizer, units, activation):# optimizer is passed because it is tuned in the parameters
    classifier = Sequential() # this is a local classifier
    classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = activation, input_dim = 55))
    classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = activation))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

def grid_search(self,X_train, y_train):
    classifier = KerasClassifier(build_fn = build_classifier) 
    parameters = {'batch_size': [10, 25, 32, 40, 100], #10
                  'epochs': [100, 500], #500
                  'optimizer': ['adam', 'rmsprop', 'sgd', 'adagrad'], #adagrad
                  'units': [5, 15, 30, 45, 60],
                  'activation': ['identity', 'logistic', 'tanh', 'relu']
                  } 
    
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10)
    
    grid_search = grid_search.fit(X_train, y_train) # fit the grid search to the data
    
    # find the attributes of the class
    best_parameters = grid_search.best_params_ 
    best_accuracy = grid_search.best_score_ 
    
    return best_parameters, best_accuracy


   
      