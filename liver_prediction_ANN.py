'''
Artificial Neural Network to predict donor-recipient matching for liver transplant

Installing Git
$ conda isntall git

Installing Theano
$ pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

Installing Tensorflow
$ pip install --user tensorflow

Installing Keras
$ pip install --upgrade keras
'''


# Importing the libraries
import numpy as np
import pandas as pd

# Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout # to prevent overfitting

# For evaluating
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score

# For tuning
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold
from tensorflow import keras
from Utility import Metric, generateMetric, generateMeanPredictions, showMetrics




# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('honours_dataset_openrefine.csv')
X = dataset.iloc[:, 1:56].values # all rows, columns index 1 to 55 (56 is excluded)
y = dataset.iloc[:, 56].values # all rows, column index 56


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)





# Part 2 - ANN building
'''
    Dropout is used to prevent overfitting. At each iteration of the training, some neurons
    are randomly disabled to prevent them from being too dependent on each other when they 
    learn the correlations so the ANN finds several independent correlations and prevents
    overfitting.
    p: the fractions os the neurons that you want to drop, 0.1 = 10%
'''

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 28, kernel_initializer = 'uniform', activation = 'relu', input_dim = 55))
classifier.add(Dropout(rate=0.1)) 

# Adding the second hidden layer
classifier.add(Dense(units = 28, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate=0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)








# Part 3 - Making predictions and evaluating the model

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # converting probabilities in the form True or False

new_prediction = classifier.predict(np.array([[50,0,31.56167151,0,0,0,0,1,0,0,0,0,0,1,0,0,165,11,15,0,0,0,1,0,0,0,0,56,1,37.109375,0,0,0,1,0,0,0,2,0,1,0.5,148,15,11,0.3,0,0,1,1,0,0,0,0,1,1]]))
# np.array to make it an array
new_prediction = (new_prediction > 0.5)

prediction_try2 = classifier.predict(np.array([[33,0,34.60207612,0,0,0,0,0,0,0,1,0,0,1,0,0,570,17,34,0,0,1,0,1,0,0,0,71,1,38.26530612,1,0,0,1,0,0,0,1,0,1,1,140,14,16,0.5,0,0,1,0,0,0,0,1,0,1]]))
prediction_try2 = (prediction_try2 > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)








# Part 4 - Evaluating, Improving and Tuning the ANN
'''
    the function builds the ANN classifier, just like in Part 2 above
    except for the fit part to the training set 
    estimator: the object to use to fit the data (classifier)
    X = the data to fit (X_train)
    y = to train a model, you need y's to understand correlations
    cv: number of folds in k-fold cross validation, 10 is recommended
    n_jobs: number of CPUs to use to do the computations, -1 means 'all CPUs'
'''

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 28, kernel_initializer = 'uniform', activation = 'relu', input_dim = 55))
    classifier.add(Dense(units = 28, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

# this line takes a while
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)

mean = accuracies.mean() # find the average of the accuracies
variance = accuracies.std() # find the variance of the accuracies (if < 1% = rather low variance)







# Part 5 - Improving and tuning the ANN
'''
    Dropout Regularization to reduce overfitting 
    PARAMETER TUNING - THE GRID SEARCH TECHNIQUE
    THIS TAKES SEVERAL HOURS LOLOLOLO
    not really but it takes a while
    
    The parameters to study
    When tuning the optimizer, they must be passed through the function
'''

def build_classifier(optimizer):# optimizer is passed because it is tuned in the parameters
    classifier = Sequential() # this is a local classifier
    classifier.add(Dense(units = 28, kernel_initializer = 'uniform', activation = 'relu', input_dim = 55))
    classifier.add(Dense(units = 28, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier) # the global classifier 
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']} 

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train) # fit the grid search to the data
best_parameters = grid_search.best_params_ # find the attributes of the class
best_accuracy = grid_search.best_score_ 