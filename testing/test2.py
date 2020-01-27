import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold
from tensorflow import keras

# importing dataset
dataset = pd.read_csv('training1.csv')
X = dataset.iloc[:, 1:56].values # all rows, columns index 1 to 55 (56 is excluded)
y = dataset.iloc[:, 56].values # all rows, column index 56
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


def neuralNetwork():
    classifier = Sequential()
    classifier.add(Dense(units = 28, kernel_initializer = 'uniform', activation = 'relu', input_dim = 55))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units = 28, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
    return classifier

classifier = KerasClassifier(build_fn = neuralNetwork, batch_size = 10, epochs = 100)
# classifier = neuralNetwork()

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()


""" TESTING """

numFeatures = X.shape[1] # find number of features (55)
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

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
