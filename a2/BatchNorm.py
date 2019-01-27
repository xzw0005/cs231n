'''
Created on Jan 25, 2019

@author: wangxing
'''
import time 
import numpy as np 
import matplotlib.pyplot as plt
from a2.utils import *

def print_mean_std(x, axis=0):
    print('\t means: ', x.mean(axis=axis))
    print('\t stds: ', x.std(axis=axis))
    print()

def rel_error(x, y):
    return np.max(np.abs(x-y) / (np.maximum(1e-8, np.abs(x)+np.abs(y))))

data = get_CIFAR10_data()
for k, v in data.items():
    print('%s: '%k, v.shape)
    
from a2.layers import *

np.random.seed(231)
N, D1, D2, D3 = 200, 50, 60, 3
W1 = np.random.randn(D1, D2)
W2 = np.random.randn(D2, D3)

bn_param = {'mode': 'train'}
gamma = np.ones((D3,))
beta = np.zeros((D3,))
for t in range(50):
    X = np.random.randn(N, D1)
    a = np. maximum(0, X.dot(W1)).dot(W2)
    batchnorm_forward(a, gamma, beta, bn_param)
bn_param['mode'] = 'test'
X = np.random.randn(N, D1)
a = np.maximum(0, X.dot(W1)).dot(W2)
print('After batch normalization (test time):')
a_norm, _ = batchnorm_forward(a, gamma, beta, bn_param)
print_mean_std(a_norm, axis=0)

np.random.seed(231)
N, D = 4, 5 
x = 5 * np.random.randn(N, D) + 12 
gamma = np.random.randn(D)
beta = np.random.randn(D)
dout = np.random.randn(N, D)
bn_param = {'mode': 'train'}
fx = lambda x: batchnorm_forward(x, gamma, beta, bn_param)[0]
fg = lambda a: batchnorm_forward(x, a, beta, bn_param)[0]
fb = lambda b: batchnorm_forward(x, gamma, b, bn_param)[0]
 
dx_numeric = eval_numerical_gradient_array(fx, x, dout)
da_numeric = eval_numerical_gradient_array(fg, gamma.copy(), dout)
db_numeric = eval_numerical_gradient_array(fb, beta.copy(), dout)
_, cache = batchnorm_forward(x, gamma, beta, bn_param)
dx, dgamma, dbeta = batchnorm_backward(dout, cache)
print('dx error: ', rel_error(dx_numeric, dx))
print('dgamma error: ', rel_error(da_numeric, dgamma))
print('dbeta error: ', rel_error(db_numeric, dbeta))

np.random.seed(231)
N, D, H1, H2, C = 2, 15, 20, 30, 10
X = np.random.randn(N, D)
y = np.random.randint(C, size=(N,))
from a2.fc_net import *
for reg in [0, 3.14]:
  print('Running check with reg = ', reg)
  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                            reg=reg, weight_scale=5e-2, dtype=np.float64,
                            normalization='batchnorm',
)
#   for name in sorted(model.params):
#     print('%s: '%name, model.params[name].shape)

  loss, grads = model.loss(X, y)
  print('Initial loss: ', loss)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
    print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
  if reg == 0: print()



