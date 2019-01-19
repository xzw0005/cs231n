'''
Created on Jan 17, 2019

@author: wangxing
'''
import numpy as np

class KNearestNeighbor(object):
    def __init__(self):
        pass
    
    def train(self, X, y):
        self.X_train = X 
        self.y_train = y 
        
    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 0:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        return self.predict_labels(dists, k=k)
    
    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            ## TODO 
            kmin = np.argsort(dists[i, :])[0:k]
            closest_y = self.y_train[kmin]
            cnts = np.bincount(closest_y)
            y_pred[i] = np.argmax(cnts)
        return y_pred
        
            
    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        ## TODO
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.sqrt(np.sum( (X[i] - self.X_train[j])**2 ) )
#                 dists[i][j] = np.linalg.norm(X[i] - self.X_train[j])
        return dists 
    
    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        ## TODO
        for i in range(num_test):
#             dists[i, :] = np.sqrt( np.sum( (X[i] - self.X_train)**2 , axis=1) )
            dists[i, :] = np.linalg.norm(X[i] - self.X_train, axis=1)
        return dists     
    
    def compute_distances_no_loops(self, X):
        sumsq_test = np.sum(np.square(X), axis=1).reshape(-1, 1) # reshape it as (500, 1)
        sumsq_train = np.sum(np.square(self.X_train), axis=1)   # shape=(5000, )
        inner_prod = np.dot(X, self.X_train.T)                  # shape=(500, 5000)
        print(sumsq_train.shape, sumsq_test.reshape(-1, 1).shape, inner_prod.shape)
        dists = sumsq_test + sumsq_train    # now shape=(500, 5000)
        dists -= 2. * inner_prod
        dists = np.sqrt(dists)
        return dists     
    

    