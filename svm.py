# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 21:36:33 2020

@author: Maria
"""

# Support Vector Regression

import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, classification_report#, multilabel_confusion_matrix
from sklearn.model_selection import GridSearchCV

###############################################
#                    SVM                      #
###############################################

class svm:
    def __init__(self, X_train, y_train, X_test, y_true):
       
        self.X_train = X_train
        self.y_train = y_train
        self.y_true = y_true
        
        #model = SVC(gamma='auto')
        model = LinearSVC(multi_class = 'crammer_singer', 
                          loss = 'squared_hinge',
                          C = 0.1,
                          random_state = 42)
           
        model.fit(X_train, y_train)
        
        # Prediction and Error
        self.predictions = model.predict(X_test)
        self.accuracy_score = accuracy_score(y_true, self.predictions) 
        self.classification_report = classification_report(y_true, self.predictions, 
                                                           output_dict = True)
        #self.multilabel_confusion_matrix = multilabel_confusion_matrix(y_true, self.predictions)
        
    def getPredictions(self):
        print("\n Predictions:", self.predictions)
        return self.predictions
    
    def getAccuracy(self):
        print("\n Accuracy Score:", self.accuracy_score)
        return self.accuracy_score
    
    def getClassificationReport(self):
        print("\n Classification report")
        for x,y in self.classification_report.items():
            print(x)
            for a, b in y.items():
                print("\t", a, " - ", b)
        return self.classification_report
    
#    def getMultilabelCM(self):
#        print(self.multilabel_confusion_matrix)
#        return self.multilabel_confusion_matrix


    ###############################################
    #                VISUALISATION                #
    ###############################################
    
    def svm_graph(self):
       plt.scatter(self.y_true, self.predictions)
       plt.xlabel('True Values')
       plt.ylabel('Predictions')
       plt.plot()
       plt.show()


    def grid_search(self):
        param_grid = { 'C': [1, 100, 0.1],
                      'max_iter': [500, 1000],
                      'loss': ['squared_hinge', 'hinge'],
                      'multi_class': ['ovr', 'crammer_singer']}  
        grid = GridSearchCV(LinearSVC(), param_grid, refit = True, verbose = 3) 
          
        # fitting the model for grid search 
        grid.fit(self.X_train, self.y_train) 
        
        print("\nBest params:", grid.best_params_)
        print("\nBest score:", grid.best_score_)
        return grid.best_params_, grid.best_estimator_, grid.best_score_
        
        




