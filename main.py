import argparse
from ast import increment_lineno
from email import parser
import matplotlib
import numpy as np
import pdb
import tensorflow as tf
import torch
import pickle
import matplotlib.pyplot as plt
from models import svm, do_pca_lreg, do_CNN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--data_num',type=int, default=1, help='Choose from 1,2,3,4,5')
args = parser.parse_args()

Data_path = "./data/Dataset" + str(args.data_num)

with open(Data_path, 'rb') as handle:
    data = pickle.load(handle)
    y = pickle.load(handle)
    testdata = pickle.load(handle)
    testy = pickle.load(handle)
    handle.close()


""" save_distribution function: for x, input is list of number of features. for label y, input is label list itself """
def save_distribution(input, isdata=True):

    if isdata:
        for i in input:
            tle = 'Frequency Histogram @ feature no.'+str(i)

            plt.hist([data[:][i], testdata[:][i]], bins=50, label=['x', 'testx'])
            plt.legend(loc='upper left')
            plt.gca().set(title=tle, ylabel='Frequency')
            plt.savefig('./images/distribution/testplot_'+str(i)+'.png', dpi=300, bbox_inches='tight')
            plt.clf()
    else: #plot label distribution
        tle = 'Frequency Histogram of label'

        plt.hist([y,testy], bins=10, label=['y', 'testy'])
        plt.legend(loc='upper left')
        plt.gca().set(title=tle, ylabel='Frequency')
        plt.savefig('./images/distribution/label.png', dpi=300, bbox_inches='tight')
        plt.clf()



# 1. Data Analysis-Data Distribution
"""
print("**Shape of train and test data: (1) train: ", data.shape,", (2) test: ", testdata.shape)
print("**Min and Max value of the label y: [", min(y),", ",max(y), ']')

save_distribution([300,500])
save_distribution(testy, isdata = False)
"""
# 1.2. Dimension Reduction and Visualization


# 2. Create Model and train

#svm(data, y, testdata, testy)

do_CNN(data, y, testdata, testy)

print("***finished***")
