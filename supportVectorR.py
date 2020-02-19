# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:42:18 2020

@author: Maria
"""

# Support Vector Regression

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

###############################################
#                    SVR                      #
###############################################

class svr:
    def __init__(self, inputs_train, output_train, inputs_test, y_test):
       
        self.y_test = y_test
        
        regr = LinearSVR(random_state=0, tol=1e-5)
           
        regr.fit(inputs_train, output_train.flatten())
                   
        # Prediction and Error
        self.predictions = regr.predict(inputs_test)
        self.errors = abs(self.predictions - y_test.flatten()) 
        self.mse = mean_squared_error(y_test.flatten(), self.predictions)
        self.mae = mean_absolute_error(y_test.flatten(), self.predictions)
        
    def getPredictions(self):
       return self.predictions
    
    def getMAE(self):
        return self.mae
    
    def getMSE(self):
        return self.mse
    
    def getMAPE(self):
        # return mean absolute percentage error (MAPE)
        return np.mean(100 * (self.errors / self.output_test.flatten()))


    ###############################################
    #                VISUALISATION                #
    ###############################################
    
    def svr_graph(self):
       plt.scatter(self.y_test, self.predictions)
       plt.xlabel('True Values [Days survived]')
       plt.ylabel('Predictions [Days survived]')
       plt.plot()
       plt.show()
