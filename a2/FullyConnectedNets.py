'''
Created on Jan 23, 2019

@author: wangxing
'''
import time 
import numpy as np
import matplotlib.pyplot as plt
from a2.utils import *
from a2.fc_net import *
from a2.layers import *
from a2.solver import Solver
  
# num_inputs = 2 
# input_shape = (4, 5, 6)
# output_dim = 3 
# input_size = num_inputs * np.prod(input_shape)
# weight_size = output_dim * np.prod(input_shape)
# 
# x = np.linspace(-.1, .5, num=input_size).reshape(num_inputs, *input_shape)
# w = np.linspace(-.2, .3, num=weight_size).reshape(np.prod(input_shape), output_dim)
# b = np.linspace(-.3, .1, num=output_dim)
# out, _ = affine_forward(x, w, b)
# correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
#                         [ 3.25553199,  3.5141327,   3.77273342]])
# 
# # Compare your output with ours. The error should be around e-9 or less.
# print('Testing affine_forward function:')
# print('difference: ', rel_error(out, correct_out))

np.random.seed(231)
N, D, H, C = 3, 5, 50, 7
X = np.random.randn(N, D)
y = np.random.randint(C, size=N)
std = 1e-3 
model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)
W1_std = abs(model.params['W1'].std() - std)
assert W1_std < std/10, '1st layer weights not right'
assert np.all(model.params['b1'] == 0), '1st layer bias not right'

model.params['W1'] = np.linspace(-.7, .3, num=D*H).reshape(D, H)
model.params['W2'] = np.linspace(-.3, .4, num=H*C).reshape(H, C)
model.params['b1'] = np.linspace(-.1, .9, num=H)
model.params['b2'] = np.linspace(-.9, .1, num=C)
X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
print(model.loss(X))
y = np.asarray([0, 5, 1])
for reg in [0., 0.7, 1.]:
    model.reg = reg 
    loss, grads = model.loss(X, y)
    print('reg = %.1f, loss = %f'%(reg, loss))
    for name in sorted(grads):
        f = lambda _: model.loss(X, y)[0]
        grad_numeric = eval_numerical_gradient(f, model.params[name], verbose=False)
        print('\t%s relative error: %.2e' % (name, rel_error(grad_numeric, grads[name])))

data = get_CIFAR10_data()
# for k, v in list(data.items()):
#     print(('%s: '% k, v.shape))
# model = TwoLayerNet(reg=0.25)
# optim_config = {'learning_rate': 1e-3}
# solver = Solver(model, data, \
#                 optim_config=optim_config, lr_decay=0.95, num_epochs=15, batch_size=256, print_every=100)
# solver.train()

num_train = 4000
small_data = {
        'X_train': data['X_train'][:num_train], 
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }
solvers = {}
for update_rule in ['sgd', 'sgd_momentum']:
    print('Running with ', update_rule)
    model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2)
    solver = Solver(model, small_data,
                    num_epochs=5, batch_size=100, update_rule=update_rule,
                    optim_config={'learning_rate': 1e-2,}, verbose=True
                    )
    solvers[update_rule] = solver 
    solver.train()
    print()
plt.subplot(3, 1, 1)
plt.title('Training loss')
plt.xlabel('Iteration')
plt.subplot(3, 1, 2)
plt.title('Training accuracy')
plt.xlabel('Epoch')
plt.subplot(3, 1, 3)
plt.title('Validation accuracy')
plt.xlabel('Epoch')
for update_rule, solver in list(solvers.items()):
    plt.subplot(3, 1, 1)
    plt.plot(solver.loss_history, 'o', label=update_rule)
    plt.subplot(3, 1, 2)
    plt.plot(solver.train_acc_history, '-o', label=update_rule)
    plt.subplot(3, 1, 3)
    plt.plot(solver.val_acc_history, '-o', label=update_rule)
for i in [1, 2, 3]:
    plt.subplot(3, 1, i)
    plt.legend(loc='upper center', ncol=4)
plt.gcf().set_size_inches(15, 15)
plt.show()












