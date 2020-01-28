# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:25:57 2020

@author: Maria
"""

# Importing the libraries
import matplotlib.pyplot as plt
import csv
import pandas
import scipy.stats as stats
import numpy as np
import pandas as pd
from scipy.stats import chisquare


# Initialising variables
(spanish_gender, spanish_diabetes, spanish_dialisis, spanish_etiology, 
 spanish_portalthrombosis, spanish_TIPStx, spanish_hepatorrenalsyndrome, 
 spanish_gender_d, spanish_diabetes_d, spanish_arterialhypertension_d,
 spanish_deathcause, spanish_hypotension, spanish_inotropes, 
 spanish_hepB, spanish_hepC, spanish_multiorgan, spanish_combinedtx, 
 spanish_completepartialgraft, spanish_ab0compatible) =([] for i in range(19))

(british_gender, british_diabetes, british_dialisis, british_etiology,
 british_portalthrombosis, british_TIPStx, british_hepatorrenalsyndrome,
 british_gender_d, british_diabetes_d, british_arterialhypertension_d, 
 british_deathcause, british_hypotension, british_inotropes, british_hepB,
 british_hepC, british_multiorgan, british_combinedtx, british_completepartialgraft,
 british_ab0compatible) = ([] for i in range(19))


# Importing the dataset
dataset = pd.read_csv('qualitative_Variables_british.csv')
dataset2 = pd.read_csv('qualitative_Variables_spanish.csv')

# Getting values from King's College Hospital
count = 0
for i in range(45):
   british_gender.append(dataset.iloc[i, 0])
   british_diabetes.append(dataset.iloc[i, 1])
   british_dialisis.append(dataset.iloc[i, 2])
   british_etiology.append(dataset.iloc[i, 3])
   british_portalthrombosis.append(dataset.iloc[i, 4])
   british_TIPStx.append(dataset.iloc[i, 5])
   british_hepatorrenalsyndrome.append(dataset.iloc[i, 6])
   british_gender_d.append(dataset.iloc[i, 7])
   british_diabetes_d.append(dataset.iloc[i, 8])
   british_arterialhypertension_d.append(dataset.iloc[i, 9])
   british_deathcause.append(dataset.iloc[i, 10])
   british_hypotension.append(dataset.iloc[i, 11])
   british_inotropes.append(dataset.iloc[i, 12])
   british_hepB.append(dataset.iloc[i, 13])
   british_hepC.append(dataset.iloc[i, 14])
   british_multiorgan.append(dataset.iloc[i, 15])
   british_combinedtx.append(dataset.iloc[i, 16])
   british_completepartialgraft.append(dataset.iloc[i, 17])
   british_ab0compatible.append(dataset.iloc[i, 18])
   count += 1
print(count)
print(len(british_gender))

# Getting values from MADR-E hospitals
count = 0
for i in range(45):
   spanish_gender.append(dataset2.iloc[i, 0])
   spanish_diabetes.append(dataset2.iloc[i, 1])
   spanish_dialisis.append(dataset2.iloc[i, 2])
   spanish_etiology.append(dataset2.iloc[i, 3])
   spanish_portalthrombosis.append(dataset2.iloc[i, 4])
   spanish_TIPStx.append(dataset2.iloc[i, 5])
   spanish_hepatorrenalsyndrome.append(dataset2.iloc[i, 6])
   spanish_gender_d.append(dataset2.iloc[i, 7])
   spanish_diabetes_d.append(dataset2.iloc[i, 8])
   spanish_arterialhypertension_d.append(dataset2.iloc[i, 9])
   spanish_deathcause.append(dataset2.iloc[i, 10])
   spanish_hypotension.append(dataset2.iloc[i, 11])
   spanish_inotropes.append(dataset2.iloc[i, 12])
   spanish_hepB.append(dataset2.iloc[i, 13])
   spanish_hepC.append(dataset2.iloc[i, 14])
   spanish_multiorgan.append(dataset2.iloc[i, 15])
   spanish_combinedtx.append(dataset2.iloc[i, 16])
   spanish_completepartialgraft.append(dataset2.iloc[i, 17])
   spanish_ab0compatible.append(dataset2.iloc[i, 18])
   count += 1
print(count)
print(len(spanish_gender))

# Function to do test on all variables
def chi_Squared_test(x):

   t, p = stats.chisquare(x)
   
   print("t = ",float(t))
   print("p = ",float(p))


chi_Squared_test(spanish_gender)









'''
def t_test(one, two):
   
   df1 = pandas.read_csv(one)
   df2 = pandas.read_csv(two)
   
   a = df1[['hillclimber']]
   b = df2[['sawtooth']]
   
   t, p = stats.ttest_ind(a,b)
   
   print("t = ",float(t))
   print("p = ",float(p))
   
t_test('hillclimber.csv', 'sawtooth.csv')




def shapiroWilkTest(dataset):
   
   df = pandas.read_csv(dataset)
   
   df1 = df[['sawtooth']]
   
   print(df1)
   
   W, P = stats.shapiro(df1)
   
   print("W = ", W)
   print("p = ",P)

shapiroWilkTest('sawtooth.csv')



def mannWhitneyTest(dataset1, dataset2):
   
   df1 = pandas.read_csv(dataset1)
   df2 = pandas.read_csv(dataset2)
   
   a = df1[['mean_squared_error']]
   b = df2[['mean_squared_error']]
   
   u, p = stats.mannwhitneyu(a, b)
   
   print(u)
   print(p)
   '''