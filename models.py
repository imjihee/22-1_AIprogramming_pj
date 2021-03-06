import os
import torch
import numpy as np
import pandas as pd
import plotly.express as px
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
import math
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
import torchvision.models as models

def svm(datax,datay,xtest, ytest, n):

    print("----------------------------------------------")
    print("***do svm for dataset", n)
    cut=round(len(datax)*0.8)
    X_t = datax[:cut]
    yt = datay[:cut]
    X_val = datax[cut:]
    yval = datay[cut:]
    X_train=np.concatenate((X_t,X_t,X_t))
    ytrain=np.concatenate((yt,yt,yt))

    loss_function = nn.CrossEntropyLoss()
    
    # For fold results
    results = {}
    # Set fixed random number seed
    torch.manual_seed(42)
    """PCA"""
    #"""
    #sc = StandardScaler()
    #X_train = sc.fit_transform(X_train)
    #xtest = sc.transform(xtest)
    #X_val = sc.transform(X_val)

    svm=SVC()
    params = {'kernel':['rbf'], 'degree':4} #poly - degree 2 or 3 // rbf - gamma 0.1 or 0.2
    print("Train start")
    #classifier=GridSearchCV(svm,params,n_jobs=2)
    svm.fit(X_train, ytrain)

    pred_val = svm.predict(X_val)
    pred = svm.predict(xtest) 
    # Process is complete.
    print('RESULT')
    acc_v=accuracy_score(pred_val, yval)
    acc_t=accuracy_score(pred, ytest)
    print("**val accuracy :", acc_v)
    print("**test accuracy : ", acc_t)
    print("----------------------------------------------")



def do_pca_lreg(xtrain, ytrain, xtest, ytest, n):

    print("----------------------------------------------")
    print("*** PCA & Logistic Regression for dataset ", n)
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
    pca = PCA(n_components = 80)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    X_val = pca.transform(X_val)

    #Fitting Logistic Regression To the training set
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, ytrain)
    #Predicting the val set result
    yv_pred = classifier.predict(X_val)
    acc_v=accuracy_score(yv_pred, yval)
    print("**val accuracy: ",acc_v)
    
    #Predicting the test set result 
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(ytest, y_pred)
    acc_t=accuracy_score(y_pred, ytest)

    print("**confusion matrix")
    print(cm)
    print("**test accuracy: ",acc_t)

    """
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
    """
    


def do_CNN(xtrain, ytrain, xtest, ytest, n):

    print("----------------------------------------------")
    print("*** CNN for dataset ", n)

    dim=math.floor(math.sqrt(len(xtrain[0])))

    if len(xtrain[0]) > dim*dim:
        xtrain=xtrain[:, :dim*dim]
        xtest=xtest[:, :dim*dim]

    #split train -> val, train
    cut=round(len(xtrain)*0.8)
    X_val = xtrain[cut:]
    yval = ytrain[cut:]
    X_train = xtrain[:cut]
    ytrain = ytrain[:cut]
    
    num_classes=max(ytrain)+1

    #normalize
    X_train = X_train / 255
    X_val = X_val / 255
    xtest = xtest / 255

    #reshape
    X_train = X_train.reshape(-1,dim,dim,1) #(num, dim, dim, 1)
    X_val = X_val.reshape(-1,dim,dim,1)
    xtest = xtest.reshape(-1,dim,dim,1)

    #ytrain=ytrain.astype(np.int64)
    #yval=yval.astype(np.int64)

    ytrain = keras.utils.to_categorical(ytrain, num_classes)
    yval = keras.utils.to_categorical(yval, num_classes)

    #create model
    model = Sequential()
    model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", 
                    input_shape=(dim,dim, 1)))
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

    #train model
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, ytrain, epochs=30, verbose=1, validation_data=(X_val, yval))

    print("==Training finished, start predict==")
    pred_v = model.predict(X_val, batch_size = 20)
    pred = model.predict(xtest, batch_size = 20)

    acc_v=accuracy_score(np.argmax(pred_v, axis=1), yval)
    acc_t=accuracy_score(np.argmax(pred, axis=1), ytest) #convert one-hot label to int (np.argmax(test, axis=1))

    print("***val accuracy: ", acc_v)
    print("***test accuracy: ", acc_t)
    print("----------------------------------------------")


def resnet(xtrain, ytrain, xtest, ytest, n):
    resnet18 = models.resnet18(pretrained=True)
    dim=math.floor(math.sqrt(len(xtrain[0])))

    if len(xtrain[0]) > dim*dim:
        xtrain=xtrain[:, :dim*dim]
        xtest=xtest[:, :dim*dim]

    #split train -> val, train
    cut=round(len(xtrain)*0.8)
    X_val = xtrain[cut:]
    yval = ytrain[cut:]
    X_train = xtrain[:cut]
    ytrain = ytrain[:cut]

    

