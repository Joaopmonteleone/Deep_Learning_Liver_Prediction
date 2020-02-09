from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import numpy as np
import pydot
from sklearn.metrics import mean_squared_error, mean_absolute_error


###############################################
#               RANDOM FOREST                 #
###############################################

class randomForest:
    def __init__(self, inputs_train, output_train, inputs_test, output_test):
        rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
        
        rf.fit(inputs_train, np.ravel(output_train))
        
        self.rf = rf
        self.feature_list = list(inputs_train.columns)
        self.inputs_train = inputs_train
        self.output_train = output_train
        self.output_test = output_test
        
        # Prediction and Error
        self.predictions = rf.predict(inputs_test)
        self.errors = abs(self.predictions - output_test.values.flatten()) 
        self.mse = mean_squared_error(output_test.values.flatten(), self.predictions)
        self.mae = mean_absolute_error(output_test.values.flatten(), self.predictions)
    
    def getMAE(self):
        return self.mae
    
    def getMSE(self):
        return self.mse
    
    def getMAPE(self):
        # return mean absolute percentage error (MAPE)
        return np.mean(100 * (self.errors / self.output_test.values.flatten()))
    
    def getImportance(self):
       # Get numerical feature importances
       importances = list(self.rf.feature_importances_)
       # List of tuples with variable and importance
       feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(self.feature_list, importances)]
       # Sort the feature importances by most important first
       feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
       # Print out the feature and importances 
       # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
       return feature_importances


    ###############################################
    #                VISUALISATION                #
    ###############################################
    def plotRandomForest(output_test, predictions):
       plt.scatter(output_test, predictions)
       plt.xlabel('True Values [Total mass]')
       plt.ylabel('Predictions [Total mass]')
       _ = plt.plot()
       plt.show()
    
    
    def makeTree(self):
       # Pull out one tree from the forest
       tree = self.rf.estimators_[5]
       # Export the image to a dot file
       export_graphviz(tree, out_file = 'images/tree.dot', feature_names = self.feature_list, rounded = True, precision = 1)
       # Use dot file to create a graph
       (graph, ) = pydot.graph_from_dot_file('images/tree.dot')
       # Write graph to a png file
       graph.write_png('images/tree.png')
    
    
       # Limit depth of tree to 3 levels
       rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
       rf_small.fit(self.inputs_train, np.ravel(self.output_train))
       # Extract the small tree
       tree_small = rf_small.estimators_[5]
       # Save the tree as a png image
       export_graphviz(tree_small, out_file = 'images/small_tree.dot', feature_names = self.feature_list, rounded = True, precision = 1)
       (graph, ) = pydot.graph_from_dot_file('images/small_tree.dot')
       graph.write_png('images/small_tree.png')