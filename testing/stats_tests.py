# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:25:57 2020

@author: Maria
"""

# Importing the libraries
import pandas
import scipy.stats as stats
import pandas as pd


# Importing the dataset
british = pd.read_csv('qualitative_Variables_british.csv')
spanish = pd.read_csv('qualitative_Variables_spanish.csv')


# Function to do test on all variables
def chi_Squared_test(x):
   
   t, p = stats.chisquare(x)
   
   print("t = ",float(t))
   print("p = ",float(p))

# Get name of columns and add to variables list
variables = []
for col in british.columns: 
    variables.append(col) 
    
variables.remove('hepB')
variables.remove('hepC')
variables.remove('ab0')
variables.remove('diabetesdon')

# For all vairables, run Chi Squared test
for var in variables:
   print(var)
   chi_Squared_test(british[[var]])

