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
import torch.nn.functional as F
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--data_num',type=int, default=1, help='Choose from 1,2,3,4,5')
parser.add_argument('--svm', action='store_true')
parser.add_argument('--lreg', action='store_true', help="logistic regression")
parser.add_argument('--cnn', action='store_true')
args = parser.parse_args()

Data_path = "./data/Dataset" + str(args.data_num)

#load data
with open("/home/esoc/jihee/AIprogramming/data/Dataset5", 'rb') as handle:
    data = pickle.load(handle)
    y = pickle.load(handle)
    testdata = pickle.load(handle)
    testy = pickle.load(handle)
    handle.close()
    pdb.set_trace()

""" save_distribution function: for x, input is list of number of features. for label y, input is label list itself """
def save_distribution(input, isdata=True):

    if isdata:
        for i in input:
            tle = 'Frequency Histogram @ feature no.'+str(i)

            plt.hist([data[:][i], testdata[:][i]], bins=50, label=['x', 'testx'])
            plt.legend(loc='upper left')
            plt.gca().set(title=tle, ylabel='Frequency')
            plt.savefig('./images/distribution/feature_dist_'+str(i)+'.png', dpi=300, bbox_inches='tight')
            plt.clf()
    else: #plot label distribution
        tle = 'Frequency Histogram of label'

        plt.hist([y,testy], bins=10, label=['y', 'testy'])
        plt.legend(loc='upper left')
        plt.gca().set(title=tle, ylabel='Frequency')
        plt.savefig('./images/distribution/label_dist.png', dpi=300, bbox_inches='tight')
        plt.clf()



# 1. Data Analysis-Data Distribution
#"""
print("**Shape of train and test data: (1) train: ", data.shape,", (2) test: ", testdata.shape)
print("**Min and Max value of the x: [", np.amin(data),", ",np.amax(data), ']')
print("**Min and Max value of the label y: [", min(y),", ",max(y), ']')

save_distribution([10, 20]) #input feature number
save_distribution(testy, isdata = False)
#"""
# 1.2. Dimension Reduction and Visualization


# 2. Create Model and train

if args.svm:
    svm(data, y, testdata, testy, args.data_num)
if args.lreg:
    do_pca_lreg(data, y, testdata, testy, args.data_num)
if args.cnn:
    do_CNN(data, y, testdata, testy, args.data_num)

print("***finished***")
