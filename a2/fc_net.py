'''
Created on Jan 23, 2019

@author: wangxing
'''
import numpy as np
from a2.layers import *
from a2.layer_utils import *

class TwoLayerNet(object):
    def __init__(self, input_dim=3*32*32, hidden_dim=100, \
                 num_classes=10, weight_scale=1e-3, reg=0.):
        self.params = {}
        self.reg = reg 
        # TODO
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)
    
    def loss(self, X, y=None):
#         scores = None
        # TODO
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        h, cache1 = affine_relu_forward(X, W1, b1)
        scores, cache2 = affine_relu_forward(h, W2, b2)
        if y is None:
            return scores 
        loss, grads = 0, {}
        # TODO
        loss, dy = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2)) 
        dh, dW2, db2 = affine_relu_backward(dy, cache2)
        dX, dW1, db1 = affine_relu_backward(dh, cache1)
        grads['W1'] = dW1 + self.reg * W1 
        grads['b1'] = db1 
        grads['W2'] = dW2 + self.reg * W2
        grads['b2'] = db2
        return loss, grads 
    
class FullyConnectedNet(object):
    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg = 0., 
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        self.normalization = normalization 
        self.use_dropout = dropout != 1 
        self.reg = reg 
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype 
        self.params = {}
        # TODO 
        dim_in = input_dim
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                dim_out = hidden_dims[i]
            else:
                dim_out = num_classes
            self.params['W%d'%(i+1)] = np.random.normal(0, weight_scale, (dim_in, dim_out))
            self.params['b%d'%(i+1)] = np.zeros(dim_out)
            if self.normalization=='batchnorm' and i < self.num_layers-1:
                self.params['gamma%d'%(i+1)] = np.ones(dim_out)
                self.params['beta%d'%(i+1)] = np.zeros(dim_out)            
            dim_in = dim_out
            
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for _ in range(self.num_layers-1)]
        for k, v in self.params.items():
            print(k, v.shape)
            self.params[k] = v.astype(dtype)
        
    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        cache_by_layer = []
        dropout_caches = {}
        h = X
        for i in range(self.num_layers-1):
            W, b = self.params['W%d'%(i+1)], self.params['b%d'%(i+1)]
            if self.normalization=='batchnorm':
                gamma = self.params['gamma%d'%(i+1)]
                beta = self.params['beta%d'%(i+1)]
                bn_param = self.bn_params[i]
                h, cache = affine_bn_relu_forward(h, W, b, gamma, beta, bn_param)
            else:
                h, cache = affine_relu_forward(h, W, b)
            cache_by_layer.append(cache)
            if self.use_dropout:
                h, dropout_caches[i+1] = dropout_forward(h, self.dropout_param)
        W = self.params['W%d'%self.num_layers]
        b = self.params['b%d'%self.num_layers]
        scores, cache = affine_forward(h, W, b)
        if mode == 'test':
            return scores 
        loss, grads = 0., {}
        loss, dout = softmax_loss(scores, y)
        
        loss += .5 * self.reg * np.sum(W * W)
        dout, dw, db = affine_backward(dout, cache)
        grads['W%d'%self.num_layers] = dw + self.reg * W 
        grads['b%d'%self.num_layers] = db
        
        for i in reversed(range(self.num_layers-1)):
            if self.use_dropout:
                dout = dropout_backward(dout, dropout_caches[i+1])
            cache = cache_by_layer.pop()
            if self.normalization=='batchnorm':
                dout, dw, db, dgamma, dbeta = affine_bn_relu_backward(dout, cache)
                grads['gamma%d'%(i+1)] = dgamma 
                grads['beta%d'%(i+1)] = dbeta
#                 print(i+1, 'bp: gamma, beta')
            else:
                dout, dw, db = affine_relu_backward(dout, cache)
            W = self.params['W%d' % (i+1)]
            loss += .5 * self.reg * np.sum(W * W)
            grads['W%d'%(i+1)] = dw + self.reg * W 
            grads['b%d'%(i+1)] = db
#             print(i+1, 'bp: W, b')
        return loss, grads
        
def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    a, fc_cache = affine_forward(x, w, b)
    bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_bn_relu_backward(dout, cache):
    fc_cache, bn_cache, relu_cache = cache 
    dbn = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward(dbn, bn_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        