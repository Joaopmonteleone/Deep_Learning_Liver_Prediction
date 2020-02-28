# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:13:53 2020

@author: Maria
"""

import numpy as np
# For the ANN building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
# For making predictions
from sklearn.metrics import confusion_matrix
# For evaluating
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
# For tuning
from sklearn.model_selection import GridSearchCV
from keras.constraints import maxnorm


class ANN:
   def __init__(self, X_train, y_train, activation_hidden, activation_output, optimizer, loss, batch_size, epochs, output_units):

       classifier = Sequential()
       classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = activation_hidden, input_dim = X_train.shape[1]))
       #classifier.add(Dropout(rate=0.1)) #EXPERIMENT WITH AND WITHOUT THIS
       classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = activation_hidden)) #relu
       #classifier.add(Dropout(rate=0.1))
       classifier.add(Dense(units = output_units, kernel_initializer = 'uniform', activation = activation_output))
       
       classifier.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
       classifier.fit(X_train, y_train, batch_size = batch_size, epochs = epochs) 
       
       self.model = classifier
    
      
   def evaluate(X_train, y_train):
       classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 500)
    
       accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
       mean = accuracies.mean()
       variance = accuracies.std()
    
       return accuracies, mean, variance

       # WITH BALANCED DATASET CATEGORICAL - IN EVALUATION
       # mean 0.3441 - very very very low
       # variance 0.08 is < 1% so good 
       
       # WITH BALANCED DATASET BINARY - 4
       # mean 0.77%
       # variance 0.07%
    
    
   def gridSearch(inputs_train, output_train):
    model = KerasClassifier(build_fn=create_model, verbose=0)

    # defining grid search parameters
    param_grid = {'optimizer': ['SGD', 'RMSprop'],  #, 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
#                  'batch_size': [10, 100, 500, 1000, 2000], 
#                  'epochs': [10, 5, 1000], 
##                  'learn_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
##                  'momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
#                  'init_mode': ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
#                  'activation': ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
#                  'weight_constraint': [1, 2, 3, 4, 5],
#                  'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#                  'neurons': [1, 5, 10, 15, 20, 25, 30]
                  }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(inputs_train, output_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return grid.best_params_, grid.best_score_


    
      
   def predict_all(self, X_pred, y_true):
       y_pred = self.model.predict(X_pred)
       y_bool = []   
       accuracy = 0
       if np.size(y_pred,1) == 1: # binary prediction (1s or 0s) predicting1 category
          print("Predicting 1 category")
          for n in y_pred:
             if n > 0.75:
                n = 1
             else:
                n = 0
             y_bool.append(n)
          # Making the Confusion Matrix
          accuracy = confusion_matrix(y_true, y_bool)
          print(accuracy)
       else: # Predicting 4 categories
          print("Predicting 4 categories")
          accuracy = multilabel_confusion_matrix(y_true, y_pred.round())
          print(accuracy)
       return y_pred, y_bool, accuracy




def build_classifier():
     classifier = Sequential()
     classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu', input_dim = 55))
     classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))
     classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
     classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
     return classifier
  

''' PARAMETER TUNING - THE GRID SEARCH TECHNIQUE
    When tuning the optimizer, the parameters to study must be passed through the function 
    Classification metrics can't handle a mix of multilabel-indicator and multiclass targets
    ONLY USE WHEN PREDICTING BINARY OUTPUTS '''

# Grid Search
def create_model(optimizer='adam',
                 #learn_rate=0.01,
                 #momentum=0,
                 init_mode='uniform',
                 activation='relu',
                 dropout_rate=0.0,
                 weight_constraint=0,
                 neurons=1
                 ):
    model = Sequential()
    model.add(Dense(neurons, 
                    input_dim=21,
                    kernel_initializer=init_mode,
                    activation=activation,
                    kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    #opimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'acc'])
    return model
    
	



   
      