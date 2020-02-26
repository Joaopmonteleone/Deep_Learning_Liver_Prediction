# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:42:18 2020

@author: Maria
"""

# Support Vector Regression

import matplotlib.pyplot as plt
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

###############################################
#                    SVR                      #
###############################################

class svr:
    def __init__(self, X_train, y_train, X_test, y_test):
       
        self.X_train = X_train
        self.y_train = y_train
        self.y_test = y_test
        
        regr = SVR(C=1.0, 
                   epsilon=0.2, 
                   gamma='scale',
                   kernel='poly'
                   )
           
        regr.fit(X_train, y_train)
                   
        # Prediction and Error
        self.predictions = regr.predict(X_test)
        self.errors = abs(self.predictions - y_test) 
        self.mse = mean_squared_error(y_test, self.predictions)
        self.mae = mean_absolute_error(y_test, self.predictions)
        
    def getPredictions(self):
        print("\nPredictions:\n",self.predictions)
        return self.predictions
    
    def getMAE(self):
        print("\nMAE:",self.mae)
        return self.mae
    
    def getMSE(self):
        print("\nMSE:",self.mse)
        return self.mse


    ###############################################
    #                VISUALISATION                #
    ###############################################
    
    def svr_graph(self):
       plt.scatter(self.y_test, self.predictions, c='#4FB5AE')
       plt.xlabel('True Values [Days survived]')
       plt.ylabel('Predictions [Days survived]')
       plt.plot()
       plt.show()
       
       
    def grid_search(self):
       param_grid = {'kernel': ['rbf','poly','sigmoid'], # poly
                     'gamma': ['scale', 'auto'], # scale
                     'C': [0.1, 1, 10, 100], # 100
                     'epsilon': [0.0001, 0.001, 0.01, 0.1, 1, 10] # 10
                     }  
       grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3) 
          
       # fitting the model for grid search 
       grid.fit(self.X_train, self.y_train) 
       
       print("\nBest params:", grid.best_params_)
       print("\nBest score:", grid.best_score_)
       return grid.best_params_
