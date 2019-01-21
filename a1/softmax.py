'''
Created on Jan 20, 2019

@author: wangxing
'''
import time
import random 
import numpy as np 
import matplotlib.pyplot as plt 
from a1.data_utils import load_CIFAR10
from a1.classifiers.linear_classifier import Softmax

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    cifar10_dir = './datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # Sub-sample
    mask = list(range(num_training, num_training+num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]
    # Preprocessing
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    # Normalize
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image
    # Bias Trick
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

def softmax_loss_naive(W, X, y, reg):
    """ W: (D, C)    X: (N, D)    y: (N,)    """
    loss = 0. 
    dW = np.zeros_like(W)
    num_train = X.shape[0] 
    for i in range(num_train):
        yhats = X[i, :].dot(W)
        yhats -= np.max(yhats)
        probs = np.exp(yhats) / np.sum(np.exp(yhats))
        loglikelihood = np.log(probs[y[i]])
        loss -= loglikelihood
        y_true = np.zeros_like(probs)
        y_true[y[i]] = 1 
        dW += np.outer(X[i, :], probs-y_true)
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W
    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    num_train = X.shape[0]
    yhats = X.dot(W)    # (N, C)
    yhats -= np.max(yhats, axis=1).reshape(num_train, -1)
    probs = np.exp(yhats)
    probs /= np.sum(probs, axis=1).reshape(num_train, -1)
    loglikelihoods = np.log(probs[np.arange(num_train), y])
    loss = -np.sum(loglikelihoods) / num_train + reg * np.sum(W * W)
    probs[range(num_train), y] -= 1 
    dW = np.dot(X.T, probs) / num_train + 2 * reg * W 
    return loss, dW 

X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
W = np.random.randn(3073, 10) * 1e-4

from a1.gradient_check import grad_check_sparse
loss, grad = softmax_loss_naive(W, X_dev, y_dev, reg=0.)
f = lambda w: softmax_loss_naive(W, X_dev, y_dev, reg=0.)[0]
grad_numeric = grad_check_sparse(f, W, grad, 10)
loss, grad = softmax_loss_naive(W, X_dev, y_dev, reg=5e1)
f = lambda w: softmax_loss_naive(W, X_dev, y_dev, reg=5e1)[0]
grad_numeric = grad_check_sparse(f, W, grad, 10)
    
tic = time.time()
loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, reg=5e-6)
toc = time.time()
print('Naive loss: %e computed in %f s' %(loss_naive, toc-tic))
tic = time.time() 
loss_vec, grad_vec = softmax_loss_vectorized(W, X_dev, y_dev, reg=5e-6)  
toc = time.time() 
print('Vectorized loss: %e computed in %f s' % (loss_vec, toc-tic)) 

grad_delta = np.linalg.norm(grad_naive - grad_vec, ord='fro')
print('Loss difference: %f' % np.abs(loss_naive-loss_vec))
print('Gradient difference: %f' % grad_delta)    
    
res = {}
best_val = -1 
best_softmax = None 
learning_rates = [1e-7, 5e-7]
regularization_strengths = [2.5e4, 5e4]

for lr in learning_rates:
    for reg in regularization_strengths:
        sm = Softmax()
        sm.train(X_train, y_train, lr=lr, reg=reg, num_iters=1500)
        train_accuracy = np.mean(sm.predict(X_train)==y_train)
        val_accuracy = np.mean(sm.predict(X_val)==y_val)
        res[(lr, reg)] = (train_accuracy, val_accuracy)
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_softmax = sm
print('best validation accuracy during cross-validation: %f'%best_val)

test_accuracy = np.mean(best_softmax.predict(X_test) == y_test)
print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))

w = best_softmax.W[:-1:] # strip out the bias
w = w.reshape(32, 32, 3, 10)
wmin, wmax = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(len(classes)):
    plt.subplot(5, 2, i+1)
    wimg = 255. * (w[:,:,:,i].squeeze()-wmin) / (wmax-wmin)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])
plt.show()