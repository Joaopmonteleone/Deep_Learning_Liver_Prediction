# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:46:19 2020

@author: Maria Luque Anguita 
"""

import pandas as pd
import tensorflow as tf
import tensorflow_docs as tfdocs
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import tensorflow_docs.modeling
import tensorflow_docs.plots


# Custom scaling function
#"""
#None of the scaling functions from within pandas allow increasing the size of
#the scaled data frame. This function takes in a float as an input which dictates
#relative to the mean of the dataset how much lower the scaling will be.
#"""
#def customScaler(listOfColumns, scale):
#    for column in listOfColumns:
#        # Select column contents by column name using [] operator
#        listOfValues = listOfColumns[column].values
#        # Increase the range of the data
#        newMin = listOfValues.mean() - ((listOfValues.mean() - listOfValues.min()) * (1 + scale))
#        newMax = listOfValues.mean() + ((listOfValues.max() - listOfValues.mean()) * (1 + scale))
#        newRange = newMax - newMin
#        scaler = MinMaxScaler()
#        scaler.fit(listOfColumns)
#        print(scaler.min_)

# Importing the dataset
dataset = pd.read_csv('datasets/regressionDataset.csv')

# Set the input and output columns
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 38].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
inputs_train, inputs_test, output_train, output_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
''' test_size is 20% = 0.2
    random_state is a generator for random sampling '''


# Normalise the data
scaler = MinMaxScaler()
inputs_train_scaled = scaler.fit_transform(inputs_train)
inputs_test_scaled = scaler.transform(inputs_test)


# Neural Network

def build_model():
  ''' This defines the structure of the neural network. the first line "layers.Dense..."
      indicates that network will take 33 inputs "input_shape(..." and have 64 outputs
      activation is just the activation function which in this case is rectified linear unit
  '''
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(inputs_train_scaled[0])]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  # The optimizer is responsible for manipulating the weights of the neural network
  # in order to achieve the desired output. The RMSprop algorithm is utilized
  optimizer = tf.keras.optimizers.RMSprop(0.001)

  # Since we want to minimize the Mean squared error to as low as possible
  # we set it to be the loss value.
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model  

model = build_model()


# Visualization of the model
plot_model(model,
           to_file='model.png',
           show_shapes=True,
           show_layer_names=True,
           rankdir='TB',
           expand_nested=True,
           dpi=96)



# How many generations do we run the algorithm
EPOCHS = 100

# Insert the training data into the model. Validation_split is allocating 20%
# of the data for the validation a.k.a not used for training. Early stop stops
# the training if there is no improvement to avoid overfitting.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
  inputs_train_scaled, output_train,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[early_stop, tfdocs.modeling.EpochDots()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

#Visualize Mean squared error over epochs
plotter = tfdocs.plots.HistoryPlotter()
plotter.plot({'Basic': history}, metric = "mse")
plt.ylim([0,100000])
plt.ylabel('MSE [Days]')

# Evaluate the model by using the test set
print("\n")
loss, mae, mse = model.evaluate(inputs_test_scaled, output_test, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} Days".format(mae))

test_predictions = model.predict(inputs_test_scaled).flatten()

a = plt.axes(aspect='equal')
plt.scatter(output_test, test_predictions)
plt.xlabel('True Values [Days]')
plt.ylabel('Predictions [Days]')
_ = plt.plot()