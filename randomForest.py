# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:29:01 2020

@author: 40011956
"""

#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import numpy as np
import pydot

###############################################
#               RANDOM FOREST                 #
###############################################

def randomForest(X_train, X_test, y_train, y_test):
   rf = RandomForestClassifier(n_estimators = 100, random_state = 42)

   rf.fit(X_train, np.ravel(y_train))

   # Predictions
   predictions = rf.predict(X_test).flatten()

   errors = abs(predictions - y_test.values.flatten())

   print('Mean Absolute Error:', round(np.mean(errors), 2))

   # Calculate mean absolute percentage error (MAPE)
   mape = 100 * (errors / y_test.values.flatten())
   # Calculate and display accuracy
   accuracy = 100 - np.mean(mape)
   print('Accuracy:', round(accuracy, 2), '%.')

   return rf, predictions


###############################################
#                VISUALISATION                #
###############################################
def plotRandomForest(output_test, predictions):
   plt.scatter(output_test, predictions)
   plt.xlabel('True Values [Total mass]')
   plt.ylabel('Predictions [Total mass]')
   _ = plt.plot()


def makeTree(rf, inputs_train, inputs_train_scaled, output_train):
   # Pull out one tree from the forest
   tree = rf.estimators_[5]
   # Export the image to a dot file
   feature_list = list(inputs_train.columns)
   export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
   # Use dot file to create a graph
   (graph, ) = pydot.graph_from_dot_file('images/tree.dot')
   # Write graph to a png file
   graph.write_png('images/tree.png')


   # Limit depth of tree to 3 levels
   rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
   rf_small.fit(inputs_train_scaled, np.ravel(output_train))
   # Extract the small tree
   tree_small = rf_small.estimators_[5]
   # Save the tree as a png image
   export_graphviz(tree_small, out_file = 'images/small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
   (graph, ) = pydot.graph_from_dot_file('images/small_tree.dot')
   graph.write_png('images/small_tree.png')
   
   return feature_list



def getImportance(rf, feature_list):
   # Get numerical feature importances
   importances = list(rf.feature_importances_)
   # List of tuples with variable and importance
   feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
   # Sort the feature importances by most important first
   feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
   # Print out the feature and importances 
   [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
   return feature_importances