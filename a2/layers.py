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

def conv_forward_naive(x, w, b, conv_param):
    """
    x: (N, C, H, W),   w: (F, C, Hf, Wf),   b: (F,)
    conv_param:  'stride', 'pad'
    out: (N, F, Ho, Wo),  cache: (x, w, b, conv_param)
    """
    stride, pad = conv_param['stride'], conv_param['pad']
    cache = (x, w, b, conv_param)
    N, C, H, W = x.shape 
    F, C, Hf, Wf = w.shape
    Ho = 1 + (H + 2 * pad - Hf) // stride 
    Wo = 1 + (W + 2 * pad - Wf) // stride
    out = np.zeros((N, F, C, Ho, Wo))
#     print(out.shape)
    paddings = [(0,0),(0,0),(pad, pad),(pad, pad)]
    xp = np.pad(x, paddings, 'constant')  # (N, C, H+2p, W+2p)
    for m in range(N):
        for l in range(F):
            for k in range(C):
                filter_k = w[l, k, :, :]    # (HF, WF)
                for j in range(Ho):
                    for i in range(Wo):
                        xf = xp[m, k, j*stride:(j*stride+Hf), i*stride:(i*stride+Wf)]
                        out[m, l, k, j, i] = np.sum(xf * filter_k)
    out = out.sum(axis=2)
    out += b.reshape(1, F, 1, 1)
    cache = (x, w, b, conv_param)
    return out, cache
    
def conv_backward_naive(dout, cache):
    x, w, b, conv_param = cache
    stride, pad = conv_param['stride'], conv_param['pad']
    db = np.sum(dout, axis=(0, 2, 3))
    N, C, H, W = x.shape 
    F, C, Hf, Wf = w.shape 
    N, C, Hout, Wout = dout.shape 
    paddings = [(0, 0), (0, 0), (pad, pad), (pad, pad)]
    xp = np.pad(x, paddings, 'constant')    # (N, C, H+2p, W+2p)
    dx = np.zeros(xp.shape)
    dw = np.zeros(w.shape)
    for m in range(N):
        for l in range(F):
            for j in range(Hout):
                for i in range(Wout):
                    xf = xp[m, :, stride*j:(stride*j+Hf), stride*i:(stride*i+Wf)]
                    dw[l, :, :, :] += dout[m, l, j, i] * xf 
                    dx[m, :, stride*j:(stride*j+Hf), stride*i:(stride*i+Wf)] += dout[m, l, j, i] * w[l, :, :, :]
    dx = dx[:, :, pad:-pad, pad:-pad]
    return dx, dw, db
    
def max_pool_forward_naive(x, pool_param):
    ph = pool_param['pool_height']
    pw = pool_param['width']
    s = pool_param['stride']
    N, C, H, W = x.shape 
    Hout = 1 + (H - ph) // s 
    Wout = 1 + (W - pw) // s 
    out = np.zeros((N, C, Hout, Wout))
    for i in range(Hout):
        for j in range(Wout):
            xpool = x[:, :, i*s:(i*s+ph), j*s:(j*s+pw)]
            out[:, :, i, j] = np.max(xpool, axis=(2, 3))
    cache = (x, pool_param)
    return out, cache 

def max_pool_backward_naive(dout, cache):
    x, pool_param = cache 
    ph = pool_param['height']
    pw = pool_param['width']
    s = pool_param['stride']
    N, C, Hout, Wout = dout.shape 
    dx = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for i in range(Hout):
                for j in range(Wout):
                    x_pool = x[n, c, i*s:(i*s+ph), j*s:(j*s+pw)]
                    o_pool = np.max(x_pool, keepdims=True)
                    dx_pool = np.zeros_like(x_pool)
                    dx_pool[x_pool == o_pool] = dout[n, c, i, j]
                    dx[n, c, i*s:(i*s+ph), j*s:(j*s+pw)] += dx_pool
    return dx
    
# x_shape = (2, 3, 4, 4)
# w_shape = (3, 3, 4, 4)
# x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
# w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
# b = np.linspace(-0.1, 0.2, num=3)
# conv_param = {'stride': 2, 'pad': 1}
# conv_forward_naive(x, w, b, conv_param)






















