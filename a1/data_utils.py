'''
Created on Jan 18, 2019

@author: wangxing
'''
import os 
from six.moves import cPickle as pickle

import random
import numpy as np

def load_CIFAR10(ROOTDIR):
    xs, ys = [], []
    for b in range(1, 6):
        f = os.path.join(ROOTDIR, 'data_batch_%d'%(b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtrain = np.concatenate(xs)
    Ytrain = np.concatenate(ys)
    del X, Y 
    Xtest, Ytest = load_CIFAR_batch(os.path.join(ROOTDIR, 'test_batch'))
    return Xtrain, Ytrain, Xtest, Ytest
        
def load_CIFAR_batch(fname):
    with open(fname, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        Y = np.array(Y)
        return X, Y 