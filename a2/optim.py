'''
Created on Jan 24, 2019

@author: wangxing
'''

import numpy as np

def sgd(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    w -= config['learning_rate'] * dw 
    return w, config

def sgd_momentum(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v 
    config['velocity'] = v 
    return next_w, config
    
def rmsprop(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))
    
    s = config.get('cache', np.zeros_like(w))
    s = config['decay_rate'] * s + (1-config['decay_rate']) * (dw * dw)
    next_w = w - config['learning_rate'] * dw / (np.sqrt(s) + config['epsilon'])
    config['cache'] = s 
    return next_w, config
    
    
    
    