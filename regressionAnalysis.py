import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
from modeling import HistoryPlotter
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error

class sequentialNN:
    def __init__(self, X_train, y_train, X_test, y_true):
        
        self.X_test = X_test
        self.y_true = y_true
        
        model = keras.Sequential([
          layers.Dense(30, activation='relu', input_shape=[len(X_train[0])]),
          layers.Dense(30, activation='relu'),
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
        # How many generations do we run the algorithm
        EPOCHS = 1000
        
        # Early stop stops the training if there is no improvement to avoid overfitting.
#        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        
        # Insert the training data into the model. Validation_split is allocating 20%
        # of the data for the validation a.k.a not used for training.
        history = model.fit(
          X_train, y_train,
          epochs=EPOCHS, validation_split = 0.2, verbose=0,
#          callbacks=[early_stop, EpochDots()]
          )
        
        self.history = history
        
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        hist.tail()
        
        self.model = model

        self.loss, self.mae, self.mse = model.evaluate(X_test, y_true, verbose=2)
    
        self.predictions = model.predict(X_test).flatten()
        
    def getPredictions(self):
       return self.predictions
        
    ###############################################
    #                VISUALISATION                #
    ###############################################
    
    def visualizeNeuralNetwork(self):
        plot_model(self.model,
           to_file='results/model.png',
           show_shapes=True,
           show_layer_names=True,
           rankdir='TB',
           expand_nested=True,
           dpi=96)
        
    def visualizeMSEoverEPOCHS(self):
        #Visualize Mean squared error over epochs
        plotter = HistoryPlotter()
        plotter.plot({'Basic': self.history}, metric = "mse")
        plt.ylim([0,100000])
        plt.ylabel('MSE [Total Mass]')
        
    def visualizePredictionsVsActual(self):        
        plt.axes(aspect='equal')
        plt.scatter(self.y_true, self.predictions, c='#FF7AA6')
        plt.xlabel('True Values [Days survived]')
        plt.ylabel('Predictions [Days survived]')
        plt.ylim([0,500])
        plt.xlim([0,500])
        plt.plot()
        plt.show()
        
    def getEvaluationMetrics(self):
        evs = explained_variance_score(self.y_true, self.predictions)
        me = max_error(self.y_true, self.predictions)
        loss = self.loss
        mae = self.mae
        mse = self.mse
        mape = np.mean(100 * (abs(self.predictions - self.y_true) / self.y_true))
        print("explained variance score:", evs, "\nme:", 
              me, "\nloss:",
              loss, "\nmae:",
              mae, "\nmse",
              mse, "\nmape",
              mape)
        return evs, me, loss, mae, mse, mape