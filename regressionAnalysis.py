import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from modeling import HistoryPlotter
from sklearn.metrics import explained_variance_score, max_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from keras.constraints import maxnorm
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


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
        # in order to achieve the desired output. The RMSprop algorithm is used
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
        self.loss, self.mae, self.mse = model.evaluate(X_test, y_true, verbose=0)
    
        self.predictions = model.predict(X_test)
        
        model.save('ann.h5')
#        print("ANN model saved to disk")
        
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
        plotter.plot({'Basic': self.history}, metric = "mse", c='#62C370')
        plt.ylim([0,100000])
        plt.ylabel('MSE [Days]')
        
    def visualizePredictionsVsActual(self):  
#        plt.scatter(self.y_true, self.predictions, c='#FF7AA6') #FF7AA6 #ECBEB4
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Precision of predicted outcomes')
        m, b = np.polyfit(self.y_true, abs(self.predictions.flatten()), 1)
        plt.plot(self.y_true, abs(self.predictions), 'o', c='#62C370')
        plt.plot(self.y_true, m*self.y_true + b) #lobf
        plt.show()
        
    def getEvaluationMetrics(self):
        evs = explained_variance_score(self.y_true, self.predictions)
        me = max_error(self.y_true, self.predictions)
        loss = self.loss
        mae = self.mae
        mse = self.mse
#        print("explained variance score:", evs, "\nme:", 
#              me, "\nloss:",
#              loss, "\nmae:",
#              mae, "\nmse",
#              mse)
        return evs, me, loss, mae, mse
    
    
    
    
def create_model(optimizer='adam',
                 #learn_rate=0.01,
                 #momentum=0,
                 init_mode='uniform',
                 activation='relu',
                 dropout_rate=0.0,
                 weight_constraint=0,
                 neurons=1
                 ):
    model = Sequential()
    model.add(Dense(neurons, 
                    input_dim=38,
                    kernel_initializer=init_mode,
                    activation=activation,
                    kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    #opimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'acc'])
    return model
    
	
def gridSearch(inputs_train, output_train):
    model = KerasClassifier(build_fn=create_model, verbose=0)

    # defining grid search parameters
    param_grid = {'optimizer': ['SGD', 'RMSprop', 'Adam' ], #best:SGD , 'Adagrad',, 'Adadelta', 'Adamax', 'Nadam'
                  'batch_size': [10, 100, 500], #best:10
                  'epochs': [100, 1000], #best:100
#                  'learn_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
#                  'momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
#                  'init_mode': ['uniform','normal'], #, 'zero', 'lecun_uniform',, 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'
#                  'activation': ['softmax','relu','sigmoid'], #, 'softplus', 'softsign', , 'tanh', , 'hard_sigmoid', 'linear'
#                 # 'weight_constraint': [1, 3, 5],
#                  'dropout_rate': [0.0, 0.9], #, 0.5
#                  'neurons': [25, 50] #10, 
                  }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=10)
    grid_result = grid.fit(inputs_train, output_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return grid.best_params_, grid.best_score_
