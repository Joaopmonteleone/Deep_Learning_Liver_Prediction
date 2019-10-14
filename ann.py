# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras






# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('honours_dataset_openrefine.csv')
X = dataset.iloc[:, 1:56].values # all rows, columns index 1 to 55 (56 is excluded)
y = dataset.iloc[:, 56].values # all rows, column index 56


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #train 80%, test 20%

'''
# Feature Scaling but not needed since already scaled
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''







# Part 2 - ANN building

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # required to initialise NN
from keras.layers import Dense # required to build layers of ANN

'''
Two ways of initialising a deep learning model:
    - defining the sequence of layers
    - defining a graph
'''

# Initialising the ANN
classifier = Sequential() #defining it as a sequence of layers
# no arguments because we define the layers step by step afterwards
'''Tip: choose the number of nodes in the hidden layer as the average of the number
of nodes in the input layer and the number of nodes in the output layer.
When the dependent variable (y) has a binary outcome (1/0) there is only one node in the
output layer.'''
# Adding the input layer AND the first hidden layer
classifier.add(Dense(units = 28, kernel_initializer = 'uniform', activation = 'relu', input_dim = 55))
'''
    units / output_dim: the number of nodes in the hidden layer (inputLayeroOfNodes (55) + outputLayerNoOfNodes (1)) / 2 = 28, dimensionality of the output space
    kernel_initializer: regulizer function applied
    activation: activation function in hidden layer (rectifier) and Sigmoid function for the output layer
    input_dim: we have 11 variables, therefore 11 input nodes
'''

# Adding the second hidden layer
classifier.add(Dense(units = 28, kernel_initializer = 'uniform', activation = 'relu'))
'''
    no need of input_dim becuase the NN already knows from last layer
    units: still applies
    same uniform method: initialises the weights randomly and gives them a small number close to 0
    activation parameter: it is recommended to be rectifier for hidden layers
    FIRST 3 LAYERS DONE: input, first and second
'''

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
'''
    units (output_dim): output layer we only want one node because dependent variable is binary
        if there are 3 categories for the dependent variable, then input 3
    kernel_initializer: keep uniform initialization method that is still used to initialise the weights
        that come from the second hidden layer
    activation: output layer requires a probability outcome a.k.a sigmoid function
        if there are 3 or more categories, change activation function to softmax: sigmoid function
        applied to a dependent variable that has more than 2 categories
'''

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
'''
    compiling an ANN: applying stochastic gradient descent on the whole ANN
    optimizer: the algorithm you wanna use to find the optimal set of weights in the NN sicne
        weights are only initialized, find weights that will make the NN the most powerful
        the stochastic gradient descent algorithm: 'adam'
    loss: the loss function within the stochastic gradient descent algorithm (the adam algorithm)
        loss function is optimized to find optimal weights
        the sum of the squared errors
        if outcome is binary: binary_crossentropy
        if more outcomes: categorical_crossentropy
    metics: criterion used to evaluate module (accuracy) used to improve model's performance
        until it reaches top accuracy
'''

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
'''
    fit ANN to the training set
    batch_size: update weights after 10 observations
    epochs: when the whole training set has been passed through the ANN
    no rule of thumb for neither of these, experiment with them to see how many are optimal

    this is where all of the processing happens
'''







# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # converting probabilities in the form True or False

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
    accuracy of 0.88: (253 + 1)/288 (288 observations in the test set)
    by changing variables maybe we can get a higher probability
'''








# Part 4 - Making predictions

new_prediction = classifier.predict(np.array([[50,0,31.56167151,0,0,0,0,1,0,0,0,0,0,1,0,0,165,11,15,0,0,0,1,0,0,0,0,56,1,37.109375,0,0,0,1,0,0,0,2,0,1,0.5,148,15,11,0.3,0,0,1,1,0,0,0,0,1,1]]))
# np.array to make it an array
new_prediction = (new_prediction > 0.5)
//correct

prediction_try2 = classifier.predict(np.array([[33,0,34.60207612,0,0,0,0,0,0,0,1,0,0,1,0,0,570,17,34,0,0,1,0,1,0,0,0,71,1,38.26530612,1,0,0,1,0,0,0,1,0,1,1,140,14,16,0.5,0,0,1,0,0,0,0,1,0,1]]))
prediction_try2 = (prediction_try2 > 0.5)
//incorrect

