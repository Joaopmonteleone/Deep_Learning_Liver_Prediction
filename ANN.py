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
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
# For tuning
from sklearn.model_selection import GridSearchCV


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
    
    
   def grid_search(X_train, y_train):
       classifier = KerasClassifier(build_fn = gridSearch_model) 
       parameters = {'batch_size': [10, 25, 32, 40, 100], #10
                     'epochs': [100, 500], #500
                     'optimizer': ['adam', 'rmsprop', 'sgd', 'adagrad'], #adagrad
                     'units': [5, 15, 30, 45, 60],
                     'activation': ['tanh', 'relu']
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
    
      
   def predict_all(self, X_pred, y_true):
       y_pred = self.model.predict(X_pred)
       y_bool = []   
       accuracy = 0
       if type(y_true) == 'numpy.ndarray': # binary prediction (1s or 0s) predicting1 category
          print("Predicting 1 category")
          for n in y_pred:
             if n > 0.75:
                n = 1
             else:
                n = 0
             y_bool.append(n)
          print(type(y_true))
          # Making the Confusion Matrix
          accuracy = confusion_matrix(y_true, y_bool)
          print(accuracy)
       else: # Predicting 4 categories
          #accuracy = accuracy_score(y_true, y_pred.round(), normalize=False)
          print()
          print("Predicting 4 categories")
          accuracy = multilabel_confusion_matrix(y_true, y_pred.round())
          print(accuracy)
          # https://scikit-learn.org/stable/modules/model_evaluation.html#multi-label-confusion-matrix
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


def gridSearch_model(optimizer, units, activation):# optimizer is passed because it is tuned in the parameters
    classifier = Sequential() # this is a local classifier
    classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = activation, input_dim = 55))
    classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = activation))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier




   
      