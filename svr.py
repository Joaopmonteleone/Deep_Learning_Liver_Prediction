# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:42:18 2020

@author: Maria
"""

# Support Vector Regression

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

###############################################
#                    SVR                      #
###############################################

class svr:
    def __init__(self, X_train, y_train, X_test, y_test):
       
        self.y_test = y_test
        
        regr = SVR(C=1.0, epsilon=0.2)
           
        regr.fit(X_train, y_train)
                   
        # Prediction and Error
        self.predictions = regr.predict(X_test)
        self.errors = abs(self.predictions - y_test) 
        self.mse = mean_squared_error(y_test, self.predictions)
        self.mae = mean_absolute_error(y_test, self.predictions)
        
    def getPredictions(self):
       return self.predictions
    
    def getMAE(self):
        return self.mae
    
    def getMSE(self):
        return self.mse
    
    def getMAPE(self):
        # return mean absolute percentage error (MAPE)
        return np.mean(100 * (self.errors / self.y_test))


    ###############################################
    #                VISUALISATION                #
    ###############################################
    
    def svr_graph(self):
       plt.scatter(self.y_test, self.predictions)
       plt.xlabel('True Values [Days survived]')
       plt.ylabel('Predictions [Days survived]')
       plt.plot()
       plt.show()
       
       
    def grid_search(self):
       param_grid = {'kernel': ['poly','rbf','sigmoid'], #,'precomputed' no gamma
                     'gamma': ['scale', 'auto'],
                     'C': ['squared_hinge', 'hinge'],
                     'multi_class': ['ovr', 'crammer_singer']}  
       grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3) 
          
       # fitting the model for grid search 
       grid.fit(self.X_train, self.y_train) 
       
       print("\nBest params:", grid.best_params_)
       print("\nBest score:", grid.best_score_)
       return grid.best_params_, grid.best_estimator_, grid.best_score_
