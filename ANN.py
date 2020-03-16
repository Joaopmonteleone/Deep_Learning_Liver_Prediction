# -*- coding: utf-8 -*-
"""
Created on Wed Jan  28 12:13:53 2020

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
    
      
   def evaluate_model(X_train, y_train):
       classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 500)
    
       accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
       mean = accuracies.mean()
       variance = accuracies.std()
    
       return accuracies, mean, variance
   
   def cross_validate(self, X_train, y_train, X_test, y_test):
       from sklearn.model_selection import StratifiedKFold
       seed = 7 # fix random seed for reproducibility
       kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
       cvscores = []
       for train, test in kfold.split(X_train, y_train):
           # create model
           model = Sequential()
           model.add(Dense(30, input_dim=X_train.shape[1], activation='relu'))
           model.add(Dense(30, activation='relu'))
           model.add(Dense(1, activation='sigmoid'))
           model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
           model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
           # evaluate the model
           scores = model.evaluate(X_test, y_test, verbose=3)
           print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
           cvscores.append(scores[1] * 100)
       print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
       return cvscores
   
   def gridSearch(inputs_train, output_train):
       model = KerasClassifier(build_fn=create_model, verbose=10)

        # defining grid search parameters
       param_grid = {'optimizer': ['RMSprop'],
                      'batch_size': [10], 
                      'epochs': [100], 
    #                  'learn_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
    #                  'momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
                      'init_mode': ['lecun_uniform'],  
                      'activation': ['softmax'],
                      'weight_constraint': [1], 
                      'dropout_rate': [0.0,0.5,0.9],
                      'neurons': [10, 30]
                      }
       grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=10)
       grid_result = grid.fit(inputs_train, output_train)
    
        # summarize results
       print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
       
    
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
                    input_dim=38,
                    kernel_initializer=init_mode,
                    activation=activation,
                    kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    #opimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'acc'])
    return model
    
	



   
      