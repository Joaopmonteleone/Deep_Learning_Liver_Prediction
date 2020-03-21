from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import numpy as np
import pydot
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import joblib

###############################################
#               RANDOM FOREST                 #
###############################################

class randomForest:
    def __init__(self, X_train, y_train, X_test, y_true, X_before):
        rf = RandomForestRegressor(n_estimators = 1000)
        
#        print("Processing Random Forest algorithm...")
        rf.fit(X_train, np.ravel(y_train))
        
        self.rf = rf
        self.feature_list = list(X_before.columns)
        self.X_train = X_train
        self.y_train = y_train
        self.y_true = y_true
        
        # Prediction and Error
        self.predictions = rf.predict(X_test)
        self.errors = abs(self.predictions - y_true) 
        self.mse = mean_squared_error(y_true.flatten(), self.predictions)
        self.mae = mean_absolute_error(y_true.flatten(), self.predictions)
        
        # save file
        filename = 'models/rf.sav'
        joblib.dump(rf, filename)
    
    def getPredictions(self):
       return self.predictions
      
    def getMAE(self):
        return self.mae
    
    def getMSE(self):
        return self.mse
    
    def getImportance(self):
       # Get numerical feature importances
       importances = list(self.rf.feature_importances_)
       # List of tuples with variable and importance
       feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(self.feature_list, importances)]
       # Sort the feature importances by most important first
       feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
       # Print out the feature and importances 
       [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
       return feature_importances


    ###############################################
    #                VISUALISATION                #
    ###############################################
    def plotRandomForest(y_true, predictions):
       plt.scatter(y_true, predictions, c='#FF7AA6') #FF7AA6 #A06CD5
       plt.xlabel('True Values')
       plt.ylabel('Predictions')
       plt.title('Precision of predicted outcomes')
       plt.plot(np.unique(y_true), np.poly1d(np.polyfit(y_true, predictions, 1))(np.unique(y_true)))
#       yerr = np.linspace(0.05, 0.2, 10)
#       plt.errorbar(y_true, predictions, yerr=yerr, label='error bar')
       plt.show()
    
    
    def makeTree(self):
       # Pull out one tree from the forest
       tree = self.rf.estimators_[5]
       # Export the image to a dot file
       export_graphviz(tree, out_file = 'results/tree.dot', feature_names = self.feature_list, rounded = True, precision = 1)
       # Use dot file to create a graph
       (graph, ) = pydot.graph_from_dot_file('results/tree.dot')
       # Write graph to a png file
       graph.write_png('results/tree.png')
    
    
       # Limit depth of tree to 3 levels
       rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
       rf_small.fit(self.X_train, np.ravel(self.y_train))
       # Extract the small tree
       tree_small = rf_small.estimators_[5]
       # Save the tree as a png image
       export_graphviz(tree_small, out_file = 'results/small_tree.dot', feature_names = self.feature_list, rounded = True, precision = 1)
       (graph, ) = pydot.graph_from_dot_file('results/small_tree.dot')
       graph.write_png('results/small_tree.png')
       
    def gridSearch(self):
       param_grid = {'n_estimators': [500, 1000], # , 2000
#                     'criterion': ['mse', 'mae'],
#                     'min_samples_split': [2, 10, 20],
#                     'min_samples_leaf': [1, 10, 100],
#                     'max_features': ['auto', 5, 'sqrt', 'log2', None],
#                     'bootstrap': [True, False],
#                     'oob_score': [True, False],
#                     'warm_start': [True, False]
                     }  
       grid = GridSearchCV(RandomForestRegressor(), param_grid, refit = True, verbose = 3) 
          
       # fitting the model for grid search 
       grid.fit(self.X_train, self.y_train) 
       
       print("\nBest params:", grid.best_params_)
       print("\nBest score:", grid.best_score_)
       return grid.best_params_
       
       