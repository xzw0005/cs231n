'''
Created on Jan 27, 2019

@author: wangxing
'''
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader 
from torch.utils.data import sampler 
import torchvision.datasets as dset 
import torchvision.transforms as T 
import numpy as np

USE_GPU = True 
dtype = torch.float32 
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Use device: ', device)
print_every = 100 

def flatten(x):
    N = x.shape[0]
    return x.view(N, -1)

def test_flatten():
    x = torch.arange(12).view(2, 1, 3, 2)
    print('Before flatten: ', x)
    print('After flatten: ', flatten(x))
test_flatten()

def two_layer_fc(x, params):
    x = flatten(x)
    w1, w2 = params 
    x = F.relu(x.mm(w1))
    x = x.mm(w2)
    return x 

def two_layer_fc_test():
    hidden_size = 42 
    x = torch.zeros((64, 50), dtype=torch.float32)
    w1 = torch.zeros((50, hidden_size), dtype=torch.float32)
    w2 = torch.zeros((hidden_size, 10), dtype=torch.float32)
    scores = two_layer_fc(x, [w1, w2])
    print(scores.size())
two_layer_fc_test()

def random_weight(shape):
    if len(shape) == 2:
        fan_in = shape[0]
    else:
        fan_in = np.prod(shape[1:])
    w = torch.randn(shape, device=device, dtype=torch.float32) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w 

def zero_weight(shape):
    return torch.zeros(shape, device=device, dtype=torch.float32, requires_grad=True)

print(random_weight((3, 5)))

def check_acc_part2(loader, model_fn, params):
    split = 'val' if loader.dataset.train else 'test'
    print('Checking accuracy on the %s set' % split)
    num_correct, num_samples = 0, 0 
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.int64)
            scores = model_fn(x, params)
            _, pred = scores.max(1)
            num_correct += (pred == y).sum()
            num_samples += pred.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100*acc))

def train_part2(model_fn, params, learning_rate):
    for t, (x, y) in enumerate(loader_train):
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)
        scores = model_fn(x, params)
        loss = F.cross_entropy(scores, y)
        loss.backward()
        with torch.no_grad():
            for w in params:
                w -= learning_rate * w.grad 
                w.grad.zero_()
        if t % print_every == 0:
            print('Iteration %d, loss = %.4f' % (t, loss.item()))
            check_acc_part2(loader_val, model_fn, params)
            print()
            
NUM_TRAIN = 49000
transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
cifar10_train = dset.CIFAR10('../datasets', train=True, \
                             download=True, transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
cifar10_val = dset.CIFAR10('../datasets', train=True, download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10('../datasets', train=False, download=True, 
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)

hidden_size = 4000
learning_rate = 1e-2 
w1 = random_weight((3*32*32, hidden_size))
w2 = random_weight((hidden_size, 10))
# train_part2(two_layer_fc, [w1, w2], learning_rate)
                

class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        nn.init.kaiming_normal_(self.fc2.weight)
    
    def forward(self, x):
        x = flatten(x)
        scores = self.fc2(F.relu(self.fc1(x)))
        return scores 

def test_TwoLayerFC():
    input_size = 50 
    x = torch.zeros((64, input_size), dtype=torch.float32) 
    model = TwoLayerFC(input_size, 42, 10)
    scores = model(x)
    print(scores.size())   
test_TwoLayerFC()

def check_acc_part34(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct, num_samples = 0, 0 
    model.eval()    # set model to evaluation mode 
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, pred = scores.max(1)
            num_correct += (pred==y).sum()
            num_samples += pred.size(0)
        acc = float(num_correct) / num_samples
        print("Got %d / %d correct (%.2f%%)" % (num_correct, num_samples, 100*acc))

def train_part34(model, optimizer, epochs=1):
    model = model.to(device=device)
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()   # set model to training mode
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_acc_part34(loader_val, model)
                print()
            
hidden_size = 4000
learning_rate = 1e-2 
model = TwoLayerFC(3*32*32, hidden_size, 10)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# train_part34(model, optimizer, epochs=1)

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)
hidden_size = 4000
learning_rate = 1e-2 
model = nn.Sequential(
        Flatten(),
        nn.Linear(3*32*32, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 10),
    )
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=0.9, nesterov=True)
train_part34(model, optimizer, epochs=1)

