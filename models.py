import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pdb
import torch.nn.functional as F
import torch
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)

def svm(datax,datay,tdatax, tdatay):

    k_folds = 10
    loss_function = nn.CrossEntropyLoss()
    
    # For fold results
    results = {}
    # Set fixed random number seed
    torch.manual_seed(42)

    kfold = KFold(n_splits=k_folds, shuffle=True) #k_folds = 10, make Kfold class

    #input_data=datax
    #target_data=datay

    svm=SVC()
    params = {'kernel':['rbf'], 'C':[10]} #poly - degree 2 or 3 // rbf - gamma 0.1 or 0.2
    print("Train start")
    classifier=GridSearchCV(svm,params,n_jobs=2)
    classifier.fit(datax, datay)

    pred = classifier.predict(tdatax)
    # Process is complete.
    print('RESULT')
    acc_t=accuracy_score(pred, tdatay)
    print("accuracy= ", acc_t)



def do_pca_lreg(xtrain, ytrain, xtest, ytest):
    cut=round(len(xtrain)*0.8)
    X_val = xtrain[cut:]
    yval = ytrain[cut:]
    X_train = xtrain[:cut]
    ytrain = ytrain[:cut]


    # performing preprocessing part
    # Doing the pre-processing part on training and testing set such as fitting the Standard scale
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(xtest)
    X_val = sc.transform(X_val)

    #Applying PCA function
    pca = PCA(n_components = 90)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    X_val = pca.transform(X_val)

    #Fitting Logistic Regression To the training set
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, ytrain)
    #Predicting the val set result
    yv_pred = classifier.predict(X_val)
    acc_v=accuracy_score(yv_pred, yval)
    print("***val accuracy: ",acc_v)
    
    #Predicting the test set result 
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(ytest, y_pred)
    acc_t=accuracy_score(y_pred, ytest)

    print("***confusion matrix")
    print(cm)
    print("***test accuracy: ",acc_t)

    
    #result through color scatter plot
    X_set, y_set = X_train, ytrain
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                        stop = X_set[:, 0].max() + 1, step = 0.01),
                        np.arange(start = X_set[:, 1].min() - 1,
                        stop = X_set[:, 1].max() + 1, step = 0.01))

    c=np.array([0.5, 0.5, 0.5]).reshape(1,-1)
    
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75
                , cmap = ListedColormap(('yellow', 'white', 'aquamarine','pink','brown','red','purple','green','orange','grey')))
    
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('yellow', 'white', 'aquamarine','pink','brown','red','purple','green','orange','grey'))(i), label = j)
    
    plt.title('Logistic Regression (Training set)')
    plt.xlabel('PC1') # for Xlabel
    plt.ylabel('PC2') # for Ylabel
    plt.legend() # to show legend
    plt.savefig('./images/svm/train.png', dpi=300, bbox_inches='tight')
    
    # show scatter plot
    plt.show()
    plt.clf()

    # Visualising the Test set results through scatter plot
    
    
    X_set, y_set = X_test, ytest
    
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                        stop = X_set[:, 0].max() + 1, step = 0.01),
                        np.arange(start = X_set[:, 1].min() - 1,
                        stop = X_set[:, 1].max() + 1, step = 0.01))
    
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75
                , cmap = ListedColormap(('yellow', 'white', 'aquamarine','pink','brown','red','purple','green','orange','grey')))
    
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('yellow', 'white', 'aquamarine','pink','brown','red','purple','green','orange','grey'))(i), label = j)
    
    # title for scatter plot
    plt.title('Logistic Regression (Test set)')
    plt.xlabel('PC1') # for Xlabel
    plt.ylabel('PC2') # for Ylabel
    plt.legend()
    plt.savefig('./images/svm/test.png', dpi=300, bbox_inches='tight')
    
    # show scatter plot
    plt.show()
    


def do_CNN(xtrain, ytrain, xtest, ytest):
    cut=round(len(xtrain)*0.8)
    X_val = xtrain[cut:]
    yval = ytrain[cut:]
    X_train = xtrain[:cut]
    ytrain = ytrain[:cut]
    num_classes=max(ytrain)+1

    X_train = X_train / 255
    X_val = X_val / 255
    xtest = xtest / 255

    X_train = X_train.reshape(-1,28,28,1)
    X_val = X_val.reshape(-1,28,28,1)
    xtest = xtest.reshape(-1,28,28,1)

    ytrain = keras.utils.to_categorical(ytrain, num_classes)
    yval = keras.utils.to_categorical(yval, num_classes)

    model = Sequential()
    model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", 
                    input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding="same"))
    model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding="same"))
    model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding="same"))
    model.add(Flatten())
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(units=num_classes, activation="softmax"))

    model.summary()

    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, ytrain, epochs=30, verbose=1, validation_data=(X_val, yval))

    print("==Training finished, start predict==")
    pred = model.predict(xtest, batch_size = 20)

    acc_t=accuracy_score(np.argmax(pred, axis=1), ytest) #convert one-hot label to int (np.argmax(test, axis=1))

    print("***test accuracy: ", acc_t)
