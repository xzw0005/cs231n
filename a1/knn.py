'''
Created on Jan 17, 2019

@author: wangxing
'''
import os
from six.moves import cPickle as pickle

import random
import numpy as np
import matplotlib.pyplot as plt

def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d'%(b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtrain = np.concatenate(xs)
    Ytrain = np.concatenate(ys)
    del X, Y 
    Xtest, Ytest = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtrain, Ytrain, Xtest, Ytest
    
def load_CIFAR_batch(fname):
    with open(fname, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        Y = datadict['labels']
        Y = np.array(Y)
        return X, Y
    
cifar10_dir = '../datasets/cifar-10-batches-py' 
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
        
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7 
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(y_train==y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(X_train[idx].astype('uint8'))
#         plt.axis('off')
#         if i==0:
#             plt.title(cls)
# plt.show()

num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
X_train = np.reshape(X_train, (X_train.shape[0], -1))
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
X_test = np.reshape(X_test, (X_test.shape[0], -1))
y_test = y_test[mask]    
print(X_train.shape, X_test.shape)

from a1.classifiers import KNearestNeighbor
import time 
knn = KNearestNeighbor.KNearestNeighbor()
knn.train(X_train, y_train)

# tic = time.time()
# dists2 = knn.compute_distances_two_loops(X_test)
# toc = time.time()
# print(dists2.shape)
# print('two loops time: %.3f' % (time.time()-tic))
#  
# k=1
# y_test_pred = knn.predict_labels(dists2, k=k)
# num_correct = np.sum(y_test_pred == y_test)
# accuracy = float(num_correct) / num_test
# print('k = %d, got %d / %d correct ==> accuracy: %.3f' %(k, num_correct, num_test, accuracy))
#  
# k=5
# y_test_pred = knn.predict_labels(dists2, k=k)
# num_correct = np.sum(y_test_pred == y_test)
# accuracy = float(num_correct) / num_test
# print('k = %d, got %d / %d correct ==> accuracy: %.3f' %(k, num_correct, num_test, accuracy))
#  
# tic = time.time()
# dists1 = knn.compute_distances_one_loop(X_test)
# # print(dists1.shape)
# print('one loop time: %.3f'%(time.time()-tic))
# diff = np.linalg.norm(dists1-dists2, ord='fro')
# if diff < 1e-3:
#     print("Oh-yeah")
# else:
#     print('Uh-oh')
# 
# tic = time.time()
# dists0 = knn.compute_distances_no_loops(X_test)
# print('one loop time: %.3f'%(time.time()-tic))
# diff = np.linalg.norm(dists0-dists2, ord='fro')
# if diff < 1e-3:
#     print("Oh-yeah")
# else:
#     print('Uh-oh')

## Cross-Validation
num_folds = 5
idx_train_folds = np.array_split(np.arange(num_training), num_folds)
print(idx_train_folds)

print(X_train.shape)
mask = np.zeros(y_train.shape, dtype=bool)
mask[idx_train_folds[0]] = True
X_validate = X_train[mask]
X_cvtrain = X_train[~mask]
# print(X_validate.shape, X_cvtrain.shape)
y_validate = y_train[mask]
y_cvtrain = y_train[~mask]
    
    
    
    