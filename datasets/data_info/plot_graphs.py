# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:29:41 2020

@author: Maria
"""

# CUMULATIVE FREQUENCY
import numpy as np
import matplotlib.pyplot as plt

# read data
data = []
with open('honours_dataset_2.csv') as csvfile:
   readCSV = csv.reader(csvfile, delimiter=',')
   for row in readCSV:
      data.append(row[39])
      
data.pop(0)
data = list(map(int, data))
      
# evaluate the histogram
values, base = np.histogram(data, bins='auto')
#evaluate the cumulative
cumulative = np.cumsum(values)
# plot the cumulative function
plt.plot(base[:-1], cumulative)

plt.show()


# CUMULATIVE FREQUENCY UP TO 365 DAYS
data2 = []
for x in data:
   if x < 360:
      data2.append(x)
values, base = np.histogram(data2, bins='auto')
cumulative = np.cumsum(values)
plt.plot(base[:-1], cumulative)
plt.show()