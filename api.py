# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 17:33:34 2020

@author: Maria
"""
from algorithms import importDataset, splitAndScale

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

output = []

def regAll():
    output.append("regAll.csv"); return ""
def regBalanced():
    output.append("regBalanced.csv"); return ""
def regEncoded():
    output.append("regEncodad.csv"); return ""
def regEncodedBalanced():
    output.append("regEncodedBalanced.csv"); return ""
def regNo365():
    output.append("regNo365.csv"); return ""
def regOnly365():
    output.append("regOnly365.csv"); return ""
def regSynthetic():
    output.append("regSynthetic.csv"); return ""
def regSyntheticWith365():
    output.append("regSyntheticWith365.csv"); return ""

def numberToDataset(argument):
    switcher = {
        1: regAll,
        2: regBalanced,
        3: regEncoded,
        4: regEncodedBalanced,
        5: regNo365,
        6: regOnly365,
        7: regSynthetic,
        8: regSyntheticWith365
    }
    dataset = switcher.get(argument, lambda: "Invalid number")
    print(dataset())
try:
    number = int(input("\nSelect number to import dataset: "))
except:
    number = int(input("\nSelect number to import dataset (1-8): "))
 
if number > 8:
    print("Number invalid")
    number = 8
    
numberToDataset(number)
print("dataset chosen:", output[0])

# Import the Dataset and separate X and y
X_before, y_before = importDataset(output[0])
# Split the dataset
X_train, X_test, y_train, y_test = splitAndScale(X_before, y_before)

print(y_test)


