'''
Artificial Neural Network to predict donor-recipient matching for liver transplant

'''


###############################################
#            IMPORTING LIBRARIES              #
###############################################



from sklearn import metrics
from sklearn.model_selection import RepeatedKFold
from tensorflow import keras
import pandas as pd
import numpy as np



###############################################
#                     PCA                     #
###############################################

# Applying PCA
from sklearn.decomposition import PCA
#pca = PCA(n_components = 2)
pca = PCA()
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
    
        
        
    
    
    
    
    
    
    
    