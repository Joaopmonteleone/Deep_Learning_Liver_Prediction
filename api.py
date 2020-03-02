# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 17:33:34 2020

@author: Maria
"""
from algorithms import importDataset, splitAndScale, ANNregression

###############################################
#              Choosing Dataset               #
###############################################
print("\n\tWELCOME TO THE LIVER TRANSPLANT DONOR-RECIPIENT MATCH PREDICTOR\n")
print("Select a file to use:")
print("1 - Regression Original 1437 rows")
print("2 - Regression Balanced (83% deleted)")
print("3 - Regression Encoded variables all 1437 rows")
print("4 - Regression Encoded variables balanced (83% deleted)")
print("5 - Regression no 365 days")
print("6 - Regression only 365 days")
print("7 - Regression only synthetic 3211 rows")
print("8 - Regression synthetic plus 365 days")

number = 0
acceptedDataset = False
while acceptedDataset is False:
    number = int(input("Select number to import dataset: "))
    if number > 0 and number < 9:
        acceptedDataset = True
    else:
        print("Invalid number, select a dataset by selecting its number (1 to 8)")

choice = ""
if number == 1: choice = "regAll.csv"
if number == 2: choice = "regBalanced.csv"
if number == 3: choice = "regEncoded.csv"
if number == 4: choice = "regEncodedBalanced.csv"
if number == 5: choice = "regNo365.csv"
if number == 6: choice = "regOnly365.csv"
if number == 7: choice = "regSynthetic.csv"
if number == 8: choice = "regSyntheticWith365.csv"

print("dataset chosen:", choice)
# Import the Dataset and separate X and y
X_before, y_before = importDataset(choice)
# Split the dataset
X_train, X_test, y_train, y_test = splitAndScale(X_before, y_before)


###############################################
#            Choosing Algorithm               #
###############################################
print("\n Choose an algorithm to run on the dataset:")
print("1 - Artificial Neural Network")
print("2 - Random Forest")
print("3 - Support Vector Regression")

number = 0
acceptedAlgorithm = False
while acceptedAlgorithm is False:
    number = int(input("Select number of algorithm to run: "))
    if number > 0 and number < 4:
        acceptedAlgorithm = True
    else:
        print("Invalid number, select an algorithm by selecting its number (1 to 3)")
        
choice = ""
if number == 1: 
    regressor = ANNregression(X_train, y_train, X_test, y_test)
if number == 2: 
    choice = "regBalanced.csv"
if number == 3: 
    choice = "regEncoded.csv"




