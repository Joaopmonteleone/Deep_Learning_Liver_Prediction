# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:25:57 2020

@author: Maria
"""

import matplotlib.pyplot as plt
import csv
import pandas
import scipy.stats as stats

spanish_gender=[]
spanish_diabetes=[]
spanish_arterialhypertension =[]
spanish_dialisis =[]
spanish_etiology =[]
spanish_portalthrombosis =[]
spanish_TIPStx =[]
spanish_hepatorrenalsyndrome =[]
spanish_upperadmsurgery = []
spanish_cmv_r = []
spanish_pretxstatus = []
spanish_gender_d = []
spanish_diabetes_d = []
spanish_arterialhypertension_d = []
spanish_deathcause = []
spanish_hypotension = []
spanish_inotropes = []
spanish_hepB = []
spanish_hepC = []
spanish_cmv_d = []
spanish_multiorgan = []
spanish_combinedtx = []
spanish_completepartialgraft = []
spanish_ab0compatible = []

british_gender=[]
british_diabetes=[]
british_arterialhypertension =[]
british_dialisis =[]
british_etiology =[]
british_portalthrombosis =[]
british_TIPStx =[]
british_hepatorrenalsyndrome =[]
british_upperadmsurgery =[]
british_cmv_r =[]
british_pretxstatus =[]
british_gender =[]
british_diabetes =[]
british_arterialhypertension =[]



import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('merged_dataset.csv')

# Getting values from King's College Hospital
count = 0
for i in range(1443):
   if dataset.iloc[i, 2] == "KC":
      british_gender.append(dataset.iloc[i, 4])
      british_diabetes.append(dataset.iloc[i, ])
      british_arterialhypertension.append(dataset.iloc[i, ])
      british_dialisis.append(dataset.iloc[i, ])
      british_etiology.append(dataset.iloc[i, ])
      british_portalthrombosis.append(dataset.iloc[i, ])
      british_TIPStx.append(dataset.iloc[i, ])
      british_hepatorrenalsyndrome.append(dataset.iloc[i, ])
      
      count += 1
print(count)
print(len(british_gender))

with open('merged_dataset.csv') as csvfile:
   readCSV = csv.reader(csvfile, delimiter=',')
   for row in readCSV:
      tournament.append(row[0])
      
with open('roulette.csv') as csvfile:
   readCSV = csv.reader(csvfile, delimiter=',')
   for row in readCSV:
      roulette.append(row[0])
      


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