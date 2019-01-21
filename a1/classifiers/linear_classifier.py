'''
Created on Jan 19, 2019

@author: wangxing
'''
import numpy as np
# from a1.classifiers.linear_svm import *

class LinearClassifier(object):
    def __init__(self):
        self.W = None 
        
    def train(self, X, y, lr=1e-3, reg=1e-5, num_iters=100, 
              batch_size=200, verbose=False):
        num_train, dim = X.shape 
        num_classes = np.max(y) + 1 
        if self.W is None:
            self.W = 1e-3 * np.random.randn(dim, num_classes)
        loss_history = []
        for it in range(num_iters):
#             X_batch = None
#             y_batch = None
            # TODO: Sample
            mask = np.random.choice(num_train, batch_size)
            X_batch = X[mask, :]
            y_batch = y[mask]
            
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            
            # TODO: update weights 
            self.W -= lr * grad
            
            if verbose and it%100 == 0:
                print('iteration %d / %d: loss %f'%(it, num_iters, loss))
        return loss_history
    
    def loss(self, X_batch, y_batch, reg):
        # subclass will override this
        pass 
    
    def predict(self, X):
#         y_pred = np.zeros(X.shape[0])
        # TODO: 
        scores = X.dot(self.W)  # (N, C)
        y_pred = np.argmax(scores, axis=1)
#         print(y_pred.shape)
        return y_pred
    
class LinearSVM(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)
        
class Softmax(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
        
def svm_loss_vectorized(W, X, y, reg):
    """ W: (D, C),    X: (N, D),    y: (N,) """
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