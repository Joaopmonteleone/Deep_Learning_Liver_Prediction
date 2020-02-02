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


###############################################
#            IMPORTING LIBRARIES              #
###############################################

# For preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# For the ANN building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
# For making predictions
from sklearn.metrics import confusion_matrix
# For evaluating
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score
# For tuning
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import RepeatedKFold
from tensorflow import keras





###############################################
#             Data Preprocessing              #
###############################################


# Importing the dataset
dataset = pd.read_csv('datasets/balanced_Dataset/export_dataframe.csv')
X = dataset.iloc[:, :-2].values # all rows, all columns except last result and 3 months answer - (1198, 39)
y_before = dataset.iloc[:, 39].values # all rows, last column (result) keep a record to compare later

# Encoding categorical data
# output NOT encoded
y = y_before
# encoding the output
onehotencoder = OneHotEncoder(categorical_features = [39])
y = onehotencoder.fit_transform(dataset.values).toarray()
y = y[:, 0:4]
# encoding the input
onehotencoder = OneHotEncoder(categorical_features = [6, 7, 14, 21, 36]) 
X = onehotencoder.fit_transform(X).toarray()
# etiology, portal thrombosis, pretransplant status performance, cause of death, cold ischemia time 

# Separating each column to predict separate classes
y_1 = y[:, 0]
y_2 = y[:, 1]
y_3 = y[:, 2]
y_4 = y[:, 3]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
''' test_size is 20% = 0.2
    random_state is a generator for random sampling '''

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
''' fit the object of the training sets and then transform it. 
    not the same for the test set, we only need to transform test set without fitting it '''





###############################################
#                ANN Building                 #
###############################################


'''
    Dropout is used to prevent overfitting. At each iteration of the training, some neurons
    are randomly disabled to prevent them from being too dependent on each other when they 
    learn the correlations so the ANN finds several independent correlations and prevents
    overfitting.
    p: the fractions of the neurons that you want to drop, 0.1 = 10%
'''


def neuralNetwork():
   # Initialising the ANN
   classifier = Sequential()
   
   # Adding the input layer and the first hidden layer
   classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu', input_dim = 55))
   classifier.add(Dropout(rate=0.1)) #EXPERIMENT WITH AND WITHOUT THIS
    
   # Adding the second hidden layer
   classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))
   classifier.add(Dropout(rate=0.1))
   
   # Adding the output layer
   classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
   
   # Compiling the ANN
   classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

   # Fitting the ANN to the Training set
   classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
 
   return classifier


#classifier = KerasClassifier(build_fn = neuralNetwork)
classifier = neuralNetwork()







###############################################
#     Make predictions & evaluate model       #
###############################################

def predict(x_test):
    return classifier.predict(x_test)

# Making new predictions from test dataset
y_pred = predict(X_test)   # percentage prediction
y_bool = []                # binary prediction (1s or 0s)

for n in y_pred:
   if n > 0.75:
      n = 1
   else:
      n = 0
   y_bool.append(n)
      
# Making the Confusion Matrix - not valid for categorical outputs, only for binary
cm = confusion_matrix(y_test, y_bool)






###############################################
#             Evaluating the ANN              #
###############################################


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu', input_dim = 55))
    classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

# this line takes a while
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
''' estimator: the object to use to fit the data (classifier)
    X = the data to fit (X_train)
    y = to train a model, you need y's to understand correlations
    cv: number of folds in k-fold cross validation, 10 is recommended
    n_jobs: number of CPUs to use to do the computations, -1 means 'all CPUs'  '''

mean = accuracies.mean() # find the average of the accuracies
variance = accuracies.std() # find the variance of the accuracies (if < 1% = rather low variance)






###############################################
#        Improving & Tuning the ANN           #
###############################################
'''
    Dropout Regularization to reduce overfitting 
    PARAMETER TUNING - THE GRID SEARCH TECHNIQUE
    When tuning the optimizer, the parameters to study must be passed through the function
'''

def build_classifier(optimizer):# optimizer is passed because it is tuned in the parameters
    classifier = Sequential() # this is a local classifier
    classifier.add(Dense(units = 28, kernel_initializer = 'uniform', activation = 'relu', input_dim = 55))
    classifier.add(Dense(units = 28, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier) # the global classifier 
parameters = {'batch_size': [10, 25, 32, 40, 50], #10
              'epochs': [100, 500, 750], #500
              'optimizer': ['adam', 'rmsprop', 'sgd', 'adagrad']} # adagrad

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train) # fit the grid search to the data

# find the attributes of the class
best_parameters = grid_search.best_params_ #[]
best_accuracy = grid_search.best_score_ 






###############################################
#                     PCA                     #
###############################################

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue', 'yellow')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue', 'yellow'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue', 'yellow')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue', 'yellow'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()



###############################################
#                Experimenting                #
###############################################

# TODO: check with https://scikit-learn.org/stable/modules/classes.html#classification-metrics

dataset_test = pd.read_csv('datasets/test1.csv')
X_test = dataset.iloc[:, 1:56].values # all rows, columns index 1 to 55 (56 is excluded)
y_test = dataset.iloc[:, 56].values

xTest_numFeatures = X_test.shape[1] # find number of features (55)
# scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
predictions = predict(X_test)

y_pred = (y_pred > 0.5)

for p in predictions:
    print(p)
    if p > 0.5:
        p = 1
        print("-")
        print (p)
    else:
        p = 0
        print("+")
        print (p)
    

metrics = precision_recall_curve(y_test, predictions)


scaler.fit(X_test)
X_test = scaler.transform(X_test)
        
        
        

times = 10

for n in range(times):
    
    print("Test Number: " + str(n))
    ANN = neuralNetwork()
    i = 0
    
    a = 0 # accuracy score
    aps = 0 # average precision score
    prc = 0 # precision_recall_curve (precision-recall pairs for different probability thresholds)
    roc = 0 # roc_curve (Receiver operating characteristic)
    bac = 0 # balanced_accuracy_score 
    
    mae = 0
    mse = 0
    rmse = 0
    
    
    kf = RepeatedKFold(n_splits=5, n_repeats=1)
    
    for trainIndex, testIndex in kf.split(X):
        
        xTrain, xTest = X[trainIndex], X[testIndex]
        yTrain, yTest = y[trainIndex], y[testIndex]
        scaler.fit(xTrain)
        xTrain = scaler.transform(xTrain)
        xTest = scaler.transform(xTest) 
    
        earlyStop = keras.callbacks.EarlyStopping(monitor = 'loss', patience = 5)
        ANN.fit(xTrain, yTrain, epochs = 100, verbose = 0, callbacks = [earlyStop])
        predictions = ANN.predict(xTest).flatten()   

        a = a + metrics.accuracy_score(yTest, predictions)
        
        mae = mae + metrics.mean_absolute_error(yTest, predictions)
        mse = mse + metrics.mean_squared_error(yTest, predictions)
        rmse = rmse + np.sqrt(metrics.mean_squared_error(yTest, predictions))
        
        i = i + 1

    a = a / i
    
    mae = mae / i
    rmse = rmse / i
    mse = mse / i
        
    print('VALIDATION RESULTS')
    print("Average Accuracy Classification Score:", a)
    
    print("Average Mean Absolute Error:", mae)
    print("Average Mean Squared Error:", mse)
    print("Average Root Mean Squared Error:", rmse)
    
        
        
    
    
    
    
    
    
    
    