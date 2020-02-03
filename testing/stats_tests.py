# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:25:57 2020

@author: Maria
"""
###############################################
#          Importing the libraries            #
###############################################

import scipy.stats as stats
import pandas as pd


###############################################
#          Importing the datasets             #
###############################################

british_classi = pd.read_csv('qualitative_Variables_british.csv')
spanish_classi = pd.read_csv('qualitative_Variables_spanish.csv')
british_numeri = pd.read_csv('quantitative_british.csv')
spanish_numeri = pd.read_csv('quantitative_spanish.csv')

###############################################
#              Chi Squared Test               #
###############################################

def chi_Squared_test(x):
   
   t, p = stats.chisquare(x)
   
   print("t = ",float(t))
   print("p = ",float(p))

# Get name of columns and add to variables list
variables = []
for col in british_classi.columns: 
    variables.append(col) 
    
variables.remove('hepB')
variables.remove('hepC')
variables.remove('ab0')
variables.remove('diabetesdon')

# For all qualitative variables, run Chi Squared test
for var in variables:
   print(var)
   #chi_Squared_test(british_classi[[var]])
   chi_Squared_test(spanish_classi[[var]])
   

###############################################
#             Shapiro Wilk Test               #
###############################################

def shapiroWilkTest(x):

   W, P = stats.shapiro(x)
   
   print("W = ", W)
   print("p = ",P)


variablesNum = []
for col in british_numeri.columns: 
    variablesNum.append(col) 
    
    
for var in variablesNum:
   print(var)
   shapiroWilkTest(british_numeri[[var]])
   shapiroWilkTest(spanish_numeri[[var]])
    
   