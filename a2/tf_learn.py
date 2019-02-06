'''
Created on Feb 5, 2019

@author: wangxing
'''
import os
import tensorflow as tf
import numpy as np
import math 
import timeit 
import matplotlib.pyplot as plt

def load_cifa10(num_training=49000, num_validation=1000, num_test=10000):
    cifar10 = tf.keras.datasets.cifar10.load_data()
    (X_train, y_train), (X_test, y_test) = cifar10
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()
    mask = range(num_training, num_training+num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    mu = X_train.mean(axis=(0,1,2), keepdims=True)
    sigma = X_train.std(axis=(0,1,2), keepdims=True)
    X_train = (X_train - mu) / sigma 
    X_val = (X_val - mu) / sigma 
    X_test = (X_test - mu) / sigma 
    return X_train, y_train, X_val, y_val, X_test, y_test
 
X_train, y_train, X_val, y_val, X_test, y_test = load_cifa10()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape, y_train.dtype)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
 
class Dataset:
    def __init__(self, X, y, batch_size, shuffle=False):
        assert X.shape[0] == y.shape[0], 'Got different #s of data and labels'
        self.X, self.y = X, y 
        self.batch_size, self.shuffle = batch_size, shuffle
         
    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size 
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B))
 
train_dset = Dataset(X_train, y_train, batch_size=64, shuffle=True)
val_dset = Dataset(X_val, y_val, batch_size=64, shuffle=True)
test_dset = Dataset(X_test, y_test, batch_size=64)
 
for t, (x, y) in enumerate(train_dset):
    print(t, x.shape, y.shape)
    if t > 5: break
     
USE_GPU = False
if USE_GPU:
    device = '/device:GPU:0'
else:
    device = '/cpu:0'
print('Use device: ', device)
print_every = 100

def flatten(x):
    N = tf.shape(x)[0]
    return tf.reshape(x, (N, -1))































    