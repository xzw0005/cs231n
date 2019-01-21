'''
Created on Jan 20, 2019

@author: wangxing
'''
import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def loss(self, X, y=None, reg=0.):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape 
        
        ## TODO
        
        scores = None
        if y is None:
            return scores 
        loss = None 
        grads = {}
        
        return loss, grads 
    
    def train(self, X, y, X_val, y_val, learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters = 100, batch_size=200, verbose=False):
        pass 
    
    def predict(self, X):
        y_pred = None 
        
        return y_pred
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        