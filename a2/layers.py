'''
Created on Jan 23, 2019

@author: wangxing
'''
import numpy as np

def affine_forward(x, w, b):
    # x: (N, d1, .., dk)     w: (D, M),    b: (M, )
    xmat = x.reshape((x.shape[0], np.prod(x.shape[1:])))   # (N, D)
    out = xmat.dot(w) + b      # (N, M)
    cache = (x, w, b)
    return out, cache 

def affine_backward(dout, cache):
    """ dout: (N, M),    x: (N, d1...dk),    w:(D, M),    b:(M, )"""
    x, w, b = cache 
    num_inputs, input_shape = x.shape[0], x.shape[1:]
    xmat = x.reshape((num_inputs, np.prod(input_shape)))    # (N, D)
    db = np.sum(dout, axis=0)
    dw = xmat.T.dot(dout)
    dx = dout.dot(w.T).reshape(num_inputs, *input_shape)
    return dx, dw, db 

def relu_forward(x):
    out = np.maximum(0, x)
    cache = x 
    return out, cache 

def relu_backward(dout, cache):
    x = cache 
    dx = dout * (x > 0)
    return dx 

def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    N, D = x.shape 
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    out, cache = None, None
    if mode == 'train':    
        mu, var = x.mean(axis=0), x.var(axis=0)
        sigma = np.sqrt(var + eps)
        xhat = (x - mu) * sigma**(-1)
        out = gamma * xhat + beta
        running_mean = momentum * running_mean + (1-momentum) * mu
        running_var = momentum * running_var + (1-momentum) * var 
        cache = (x, gamma, mu, sigma)
    elif mode == 'test':
        sigma = np.sqrt(running_var + eps)
        xhat = (x - running_mean) * sigma**(-1)
        out = gamma * xhat + beta 
    else:
        raise ValueError('Invalid forward batch normalization mode "%s"' % mode)
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var    
    return out, cache

def batchnorm_backward(dout, cache):
    x, gamma, mu, sigma = cache
    N = x.shape[0]
    dxhat = dout * gamma 
    dvar = np.sum(dxhat * (x-mu) * (-1./2) * sigma**(-3), axis=0)
    dmu = np.sum(-dxhat/sigma, axis=0) + dvar * np.sum(-2.*(x-mu), axis=0)/N
    dx = dxhat / sigma + dvar * 2. * (x-mu)/N + dmu / N
    dgamma = np.sum(dout * (x-mu)/sigma, axis=0)
    dbeta = np.sum(dout, axis=0)
    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    p, mode = dropout_param['p'], dropout_param['mode']
    mask = None 
    out = x
    if mode == 'train':
        if 'seed' in dropout_param:
            np.random.seed(dropout_param['seed'])
        mask = np.random.rand(*x.shape) < p 
        out *= (mask / p) 
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache 

def dropout_backward(dout, cache):
    dropout_param, mask = cache 
    mode = dropout_param['mode']
    dx = None
    if mode == 'train':
        p = dropout_param['p']
        dx = dout * (mask / p)
    elif mode == 'test':
        dx = dout
    return dx 

def softmax_loss(yhat, y):
    N = yhat.shape[0]
    logits = yhat - np.max(yhat, axis=1, keepdims=True)
    Z = np.sum(np.exp(logits), axis=1, keepdims=True)
    logprobs = logits - np.log(Z)
    loss = -np.sum(logprobs[np.arange(N), y]) / N
    
    dout = np.exp(logprobs)
    dout[np.arange(N), y] -= 1 
    dout /= N 
    return loss, dout

























