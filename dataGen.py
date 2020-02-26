# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:16:40 2020

@author: Maria
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
import random

dataset = pd.read_csv('datasets/regNo365.csv')

# make a list of all column names    
columnNames = []
for col in dataset.columns:
    columnNames.append(col)

###############################################
#                  Random                     #
###############################################
def makeRandomData():
    # make a dataframe of age so the other columns can be added later on
    age = pd.DataFrame(np.random.randint(15,77,size=(8000, 1)), columns=list('a'))
    age = age.rename(columns={'a':'age'})
    global concatenated
    columnNames.remove('age')
    # for every column, create 8000 rows of random numbers between that column's max and min values
    for col in columnNames:
        df = pd.DataFrame(np.random.randint((dataset[col].min()),(dataset[col].max()+1),size=(8000, 1)), columns=list('a'))
        df = df.rename(columns={'a':col}) # rename column name to what it is
        age = pd.concat([age, df], axis=1) # join lists
    newdataset = dataset.append(age, ignore_index = True, sort=False) # add synthetic data to original data
    newdataset.to_csv(r'C:\Users\Maria\Desktop\Deep_Learning_Liver_Prediction\datasets\syntheticDataReg.csv',
                   index = False)


###############################################
#               Slight Mutations              #
###############################################
def slightMutations():
    print("Slight mutations function")
    
    Row_list =[] 
    for index, rows in dataset.iterrows(): 
        # Create list for the current row 
        my_list = []
        for col in columnNames:
            my_list.append(rows[col]) 
        # append the list to the final list 
        Row_list.append(my_list) 
  
    for row in Row_list:
        for index in range(dataset.shape[1]):
            print(index)
            print("col name:", columnNames[index])
            mutate(row, index)
            print()
        print("\n---New row---\n")
        
    
        
slightMutations()
    
def mutate(value, index):
    print("before:",int(value[index]))
    rangeMax = dataset[columnNames[index]].max()
    rangeMin = dataset[columnNames[index]].min()
    rangeTotal = rangeMax - rangeMin
#    print("Max:", rangeMax)
#    print("Min:", rangeMin)
#    print("rangeTotal:", rangeTotal)
    if rangeTotal == 1: # if binary outcome
        #print("---BINARY")
        if uniform(0,1) < 0.5:
            #print("------CHANGED")
            value[index] = int(value[index]) ^ 1 # flip 1s to 0s and 0s to 1s
    else: # non-binary outcome
        percentage = int((5 * rangeTotal) / 100.0)
        randomNo = random.randint(-percentage, percentage)
#        print("random:", randomNo)
        value[index] = int(abs(value[index] + randomNo))
    print("after:", int(value[index]))
    #return 



from random import randint, uniform;

def mutateIndividual(ind):
     mutationIndex = randint(0, len(ind)) # select one chromosome
     ind[mutationIndex] = randint(0,3) # inclusive both
     return ind;

for i in range(0, len(population)): # outer loop on each individual
     population[i] = mutateIndividual(population[i]);
    
  