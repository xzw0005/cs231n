'''
Created on Jan 20, 2019

@author: wangxing
'''
import numpy as np
import random 

def eval_numerical_gradient(f, x, verbose=True, h=1e-5):
#     fx = f(x)
    grad = np.zeros_like(x)
    # iterate over all indices in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['rw'])
    while not it.finished:
        ix = it.multi_index
        old_val = x[ix]
        x[ix] = old_val + h 
        fxph = f(x)
        x[ix] = old_val - h 
        fxmh = f(x)
        x[ix] = old_val
        grad[ix] = (fxph - fxmh) / (2 * h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()
    return grad

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags='rw')
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h 
        pos = f(x).copy() 
        x[ix] = oldval - h 
        neg = f(x).copy() 
        x[ix] = oldval
        grad[ix] = np.sum((pos-neg) * df) / (2 * h)
        it.iternext()
    return grad

def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    for i in range(num_checks):
        ix = tuple([random.randrange(m) for m in x.shape])
        oldval = x[ix]
        x[ix] = oldval + h 
        fxph = f(x) 
        x[ix] = oldval - h 
        fxmh = f(x)
        x[ix] = oldval 
        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))
