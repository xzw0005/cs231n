'''
Created on Jan 23, 2019

@author: wangxing
'''

import os 
from six.moves import cPickle as pickle

import random
import numpy as np

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, subtract_mean=True):
    cifar10_dir = '../datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    mask = list(range(num_training, num_training+num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()
    return {'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test}

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

def rel_error(x, y):
    return np.max(np.abs(x-y) / np.maximum(1e-8, np.abs(x)+np.abs(y)))

def eval_numerical_gradient(f, x, verbose=True, h=1e-5):
#     fx = f(x)
    grad = np.zeros_like(x)
    # iterate over all indices in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_val = x[ix]
        x[ix] = old_val + h 
        fxph = f(x)
        x[ix] = old_val - h 
        fxmh = f(x)
        x[ix] = old_val
        grad[ix] = (fxph - fxmh) / (2 * h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()
    return grad

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h 
        pos = f(x).copy() 
        x[ix] = oldval - h 
        neg = f(x).copy() 
        x[ix] = oldval
        grad[ix] = np.sum((pos-neg) * df) / (2 * h)
        it.iternext()
    return grad

def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    for i in range(num_checks):
        ix = tuple([random.randrange(m) for m in x.shape])
        oldval = x[ix]
        x[ix] = oldval + h 
        fxph = f(x) 
        x[ix] = oldval - h 
        fxmh = f(x)
        x[ix] = oldval 
        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))