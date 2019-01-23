'''
Created on Jan 20, 2019

@author: wangxing
'''
import numpy as np

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
        N = X.shape[0] 
        
        h = np.maximum(X.dot(W1) + b1, 0)   # (N, H)
        scores = h.dot(W2) + b2         # (N, C)
        if y is None:
            return scores 
        yhats = scores - np.max(scores, axis=1, keepdims=True)
        probs = np.exp(yhats)
        probs /= np.sum(probs, axis=1, keepdims=True)
        loglkhd = np.log(probs[np.arange(N), y])
        loss = np.sum(-loglkhd) / N + reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        
        grads = {}
        dy = probs 
        dy[np.arange(N), y] -= 1    # (N, C)
        dy /= N     # !!!!!!
        dW2 = np.dot(h.T, dy)   # (H, C) = (H, N) x (N, C)
        db2 = np.sum(dy, axis=0) # (C, 1)
        
        dh = np.dot(dy, W2.T)   # (N, H) = (N, C) x (C, H)
        dh = dh * (h > 0)
        dW1 = np.dot(X.T, dh)      # (D, H) = (D, N) x (N, H)
        db1 = np.sum(dh, axis=0)
        
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        
        grads['W1'] += 2*reg*W1 
        grads['W2'] += 2 * reg * W2
        return loss, grads 
    
    def train(self, X, y, X_val, y_val, learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters = 100, batch_size=200, verbose=False):
        num_train = X.shape[0]
        iters_per_epoch = max(num_train / batch_size, 1)
        
        loss_history, train_acc_history, val_acc_history = [], [], []
        for it in range(num_iters):
            mask = np.random.choice(num_train, batch_size)
            X_batch = X[mask, :]
            y_batch = y[mask]
            
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)
            
            for pname in self.params:
                self.params[pname] -= learning_rate * grads[pname]
            
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
            if it % iters_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                train_acc_history.append(train_acc)
                val_acc = (self.predict(X_val) == y_val).mean()
                val_acc_history.append(val_acc)
                
                learning_rate *= learning_rate_decay
                
        return {'loss_history': loss_history, 
                'train_acc_history': train_acc_history, 
                'val_acc_history': val_acc_history}
    
    def predict(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        h = np.maximum(X.dot(W1) + b1, 0)
        o = h.dot(W2) + b2
        y_pred = np.argmax(o, axis=1)
        return y_pred