'''
Created on Jan 21, 2019

@author: wangxing
'''
import numpy as np
import matplotlib.pyplot as plt
from a1.classifiers.neural_net import TwoLayerNet

input_size = 4 
hidden_size = 10 
num_classes = 3
num_inputs = 5

def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0,1,2,2,1])
    return X, y 

net = init_toy_model()
X, y = init_toy_data()
scores = net.loss(X)
print(scores)
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print('Difference between your scores and correct scores:', np.sum(np.abs(scores-correct_scores)))

from a1.gradient_check import eval_numerical_gradient

loss, grads = net.loss(X, y, reg=0.05)
correct_loss = 1.30378789133
print('Difference between your loss and correct loss:', np.sum(np.abs(loss-correct_loss)))

def relative_error(x, y):
    return np.max(np.abs(x-y) / np.maximum(1e-8, np.abs(x)+np.abs(y)))

for param_name in grads:
    f = lambda W: net.loss(X, y, reg=0.05)[0]
    param_grad_numeric = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    print('%s max relative error: %e' % (param_name, relative_error(param_grad_numeric, grads[param_name])))

## Train the network
net = init_toy_model()
stats = net.train(X, y, X, y, learning_rate=1e-1, reg=5e-6, num_iters=100, verbose=False)
print('Final training loss: ', stats['loss_history'][-1])
# plt.plot(stats['loss_history'])
# plt.xlabel('iteration')
# plt.ylabel('training loss')
# plt.title('Training loss history')
# plt.show()

from a1.data_utils import load_CIFAR10
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    cifar10_dir = '../datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10 
net = TwoLayerNet(input_size, hidden_size, num_classes)
stats = net.train(X_train, y_train, X_val, y_val, num_iters=1000, batch_size=200,
                  learning_rate=1e-4, learning_rate_decay=0.95, reg=0.25, verbose=True)
val_acc = np.mean(net.predict(X_val) == y_val)
print('Validation accuracy: ', val_acc)

# # Debug strategy # 1: Plot loss function
# plt.subplot(2, 1, 1)
# plt.plot(stats['loss_history'])
# plt.title('Loss History')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.subplot(2, 1, 2)
# plt.plot(stats['train_acc_history'], label='Train')
# plt.plot(stats['val_acc_history'], label='Validation')
# plt.title('Classification accuracy history')
# plt.xlabel('Epoch') 
# plt.ylabel('Classification Accuracy')
# plt.legend()
# plt.show()

# # Debug strategy # 2: Visualize weights of 1st layer
# from a1.vis_utils import visualize_grid
# def show_net_weights(net):
#     W1 = net.params['W1']
#     W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
#     plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
#     plt.gca().axis('off')
#     plt.show()
# show_net_weights(net)
