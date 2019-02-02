'''
Created on Feb 1, 2019

@author: wangxing
'''
import numpy as np
from a2.layer import *
from a2.layer_utils import *

class ThreeLayerConvNet(object):
    '''
    conv -> relu -> 2x2 max pool 
    -> affine -> relu - affine -> softmax
    '''


    def __init__(self, input_dim=(3, 32, 32), num_filters=32,
                 filter_size=7, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0., dtype=np.float32):
        self.params = {} 
        self.reg = reg 
        self.dtype = dtype 
        
        num_channels = input_dim[0]
        w1_shape = (num_filters, num_channels, filter_size, filter_size)
        self.params['W1'] = np.random.normal(0, weight_scale, w1_shape)
        self.params['b1'] = np.zeros(num_filters)
        # According to conv_param & pool_param given below, same padding with output shape same as input 
        conv_out_size = np.prod(w1_shape)
        self.params['W2'] = np.random.normal(0, weight_scale, (conv_out_size, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
            
    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        filter_size = W1.shape[2]
        conv_param = {'stride':1, 'pad':(filter_size-1)//2}
        pool_param = {'pool_height':2, 'pool_width':2, 'stride':2}
        ## Forward Pass 
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out2, cache2 = affine_relu_forward(out1, W2, b2)
        out3, cache3 = affine_forward(out2, W3, b3)
        scores = out3
        if y is None:
            return scores 
        grads = {} 
        loss, dout = softmax_loss(scores, y)
        dout, grads['W3'], grads['b3'] = affine_backward(dout, cache3)
        loss += 0.5 * self.reg * np.square(W3)
        grads['W3'] += self.reg * W3
        dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, cache2)
        loss += 0.5 * self.reg * np.square(W2)
        grads['W2'] += self.reg * W2 
        dout, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout, cache1)
        loss += 0.5 * self.reg * np.square(W1)
        grads['W1'] += self.reg * W1
        return loss, grads

























        
        