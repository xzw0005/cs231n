'''
Created on Jan 18, 2019

@author: wangxing
'''
import random 
import numpy as np 

def svm_loss_naive(W, X, y, reg):
    """ W: (D, C),    X: (N, D),    y: (N,) """
    dW = np.zeros(W.shape)
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0 
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 
            if margin > 0:
                loss += margin 
                dW[j, :] += X[i, :]
                dW[y[i], :] -= X[i, :]
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W    
    return loss, dW
    
def svm_loss_vectorized(W, X, y, reg):
    """ W: (D, C),    X: (N, D),    y: (N,) """
    dW = np.zeros(W.shape)
    scores = X.dot(W)   # (N, C)
    num_train = X.shape[0]
    correct_class_scores = scores[np.arange(num_train), y]  # (N, )
    correct_class_scores = correct_class_scores.reshape((num_train, -1))    # (N, 1)
    scores += (1 - correct_class_scores)
    scores[np.arange(num_train), y] = 0.
    scores = np.maximum(0., scores)
    loss = np.sum(scores) / num_train + reg * np.sum(W * W)
    
    scores[scores > 0] = 1
    cnts = np.sum(scores, axis=1)
    scores[np.arange(num_train), y] = -cnts
    dW = np.dot(X.T, scores) / num_train
    dW += 2 * reg * W
    
    return loss, dW
    
