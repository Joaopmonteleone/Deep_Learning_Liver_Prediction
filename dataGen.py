# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:16:40 2020

@author: Maria
"""
import numpy as np
import pandas as pd
import random

dataset = pd.read_csv('datasets/regAll.csv')
dataset2 = pd.read_csv('datasets/regSynthetic.csv')
regNo365 = pd.read_csv('datasets/regOnly365.csv')

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
def mutate(value, index):
    rangeMax = regNo365[columnNames[index]].max() # find max value of column
    rangeMin = regNo365[columnNames[index]].min() # find min value of column
    rangeTotal = rangeMax - rangeMin # find the range of values
    if rangeTotal == 1: # if binary value
        if random.uniform(0,1) < 0.5: # 50% chance of flipping bit
            value[index] = int(value[index]) ^ 1 # flip 1s to 0s and 0s to 1s
    else: # non-binary outcome
        percentage = int((5 * rangeTotal) / 100.0) # change value 5% of its range
        randomNo = random.randint(-percentage, percentage) # create a random number in the 5% closest to that number
        value[index] = int(abs(value[index] + randomNo)) # make the number always positive so it is valid (absolute value)
    return value[index]
    
newDataset = [] # new created data will be added to this list

def slightMutations():
    # transform dataset dataframe to list to iterate through it
    dataset_list = regNo365.values.tolist()
    for row in dataset_list:
        newRow = [] # new observation created
        for index in range(regNo365.shape[1]):
            newValue = mutate(row, index)
            newRow.append(newValue)
        newDataset.append(newRow)
   
for i in range(5): # run algorithm 5 times, creates 214 different rows each time
    slightMutations()
    
newDataset = pd.DataFrame(newDataset) # convert to pandas dataframe
newDataset.columns = columnNames # rename columns



export_csv = together.to_csv (r'C:\Users\Maria\Desktop\Deep_Learning_Liver_Prediction\datasets\regSyntheticWith365.csv',
                             index = None, 
                             header=True) 

together = pd.concat([newDataset, regNo365]) # join lists


