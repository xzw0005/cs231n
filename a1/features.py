'''
Created on Jan 22, 2019

@author: wangxing
'''
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from a1.data_utils import load_CIFAR10
from scipy.ndimage.filters import uniform_filter
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    cifar10_dir = './datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    mask = list(range(num_training, num_training+num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()    

def color_histogram_hsv(img, nbin=10, xmin=0, xmax=255, normalized=True):
    ndim = img.ndim 
    bins = np.linspace(xmin, xmax, nbin+1)
    hsv = matplotlib.colors.rgb_to_hsv(img/xmax) * xmax
    img_hist, bin_edges = np.histogram(hsv[:,:,0], bins=bins, density=normalized)
    img_hist = img_hist * np.diff(bin_edges)
    return img_hist

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def hog_feature(img):
    if img.ndim == 3:
        image = rgb2gray(img)
    else:
        image = np.at_least_2d(img)
    sx, sy = image.shape 
    orientations = 9
    cx, cy = (8, 8)
    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)
    gy[:-1, :] = np.diff(image, n=1, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_ori = np.arctan2(gy, (gx+1e-15)) * (180/np.pi) + 90
    n_cellsx = int(np.floor(sx/cx))
    n_cellsy = int(np.floor(sy/cy))
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        temp_ori = np.where(grad_ori < 180 / orientations * (i+1), grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i, temp_ori, 0)
        cond2 = temp_ori > 0 
        temp_mag = np.where(cond2, grad_mag, 0)
        orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[int(cx/2)::cx, int(cy/2)::cy].T
    return orientation_histogram.ravel()

def extract_features(imgs, feature_fns, verbose=False):
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, 'Feature functions must be 1-dimensional'
        feature_dims.append(feats.size)
        first_image_features.append(feats)
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T 
    for i in range(1, num_images):
        idx = 0 
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i%1000==0:
            print('Done extracting features for %d / %d images' % (i, num_images))
    return imgs_features

num_color_bins = 10
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

std_feats = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feats
X_val_feats /= std_feats
X_test_feats /= std_feats

X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])

# Train SVM on features
from a1.classifiers.linear_classifier import LinearSVM
learning_rates = [1e-9, 1e-8, 1e-7]
regularization_strengths = [5e4, 5e5, 5e6]
results = {}
best_val = -1 
best_svm = None 
for lr in learning_rates:
    for reg in regularization_strengths:
        svm = LinearSVM()
        svm.train(X_train_feats, y_train, lr, reg, num_iters=1500)
        y_train_pred = svm.predict(X_train_feats)
        train_accuracy = np.mean(y_train_pred == y_train)
        y_val_pred = svm.predict(X_val_feats)
        val_accuracy = np.mean(y_val_pred == y_val)
        results[(lr, reg)] = (train_accuracy, val_accuracy)
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_svm = svm 
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy))
y_test_pred = best_svm.predict(X_test_feats)
test_accuracy = np.mean(y_test_pred == y_test)
print(test_accuracy)
examples_per_class = 8 
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for cls, cls_name in enumerate(classes):
    idxs = np.where((y_test != cls) & (y_test_pred==cls))[0]
    idxs = np.random.choice(idxs, examples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(examples_per_class, len(classes), i*len(classes)+cls+1)
        plt.imshow(X_test[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls_name)
plt.show()


















