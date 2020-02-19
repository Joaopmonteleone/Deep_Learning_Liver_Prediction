# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 21:36:33 2020

@author: Maria
"""

# Support Vector Regression

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report #, multilabel_confusion_matrix

###############################################
#                    SVM                      #
###############################################

class svm:
    def __init__(self, X_train, y_train, X_test, y_true):
       
        self.y_true = y_true
        
        model = SVC(random_state=0, tol=1e-5)
           
        model.fit(X_train, y_train.flatten())
        
        # Prediction and Error
        self.predictions = model.predict(X_test)
        self.accuracy_score = accuracy_score(y_true, self.predictions) 
        target_names = ['class 1', 'class 2', 'class 3', 'class 4']
        self.classification_report = classification_report(y_true, self.predictions, 
                                                           target_names = target_names,
                                                           output_dict = True)
        #self.multilabel_confusion_matrix = multilabel_confusion_matrix(y_true, self.predictions)
        
    def getPredictions(self):
        print(self.predictions)
        return self.predictions
    
    def getAccuracy(self):
        print(self.accuracy_score)
        return self.accuracy_score
    
    def getClassificationReport(self):
        for x,y in self.classification_report.items():
            print(x)
            for a, b in y.items():
                print("\t", a, " - ", b)
            
        return self.classification_report
    
#    def getMultilabelCM(self):
#        # return mean absolute percentage error (MAPE)
#        return self.multilabel_confusion_matrix


    ###############################################
    #                VISUALISATION                #
    ###############################################
    
    def svm_graph(y_true, y_pred):
       plt.scatter(y_true, y_pred)
       plt.xlabel('True Values')
       plt.ylabel('Predictions')
       _ = plt.plot()
       plt.show()







