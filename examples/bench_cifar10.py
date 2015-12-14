# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

import sys
import pickle

import numpy as np

PRETRAIN = False


def load(name):
    # Pickle module isn't backwards compatible. Hack so it works:
    compat = {'encoding': 'latin1'} if sys.version_info[0] == 3 else {}

    print("\t"+name)
    try:
        with open(name, 'rb') as f:
            return pickle.load(f, **compat)
    except IOError:
        import gzip
        with gzip.open(name+'.gz', 'rb') as f:
            return pickle.load(f, **compat)


# Download and extract Python data for CIFAR10 manually from here:
#     http://www.cs.toronto.edu/~kriz/cifar.html

print("Loading...")
dataset1 = load('data_batch_1')
dataset2 = load('data_batch_2')
dataset3 = load('data_batch_3')
dataset4 = load('data_batch_4')
dataset5 = load('data_batch_5')
dataset0 = load('test_batch')
print("")

data_train = np.vstack([dataset1['data']]) #, dataset2['data'], dataset3['data'], dataset4['data'], dataset5['data']])
labels_train = np.hstack([dataset1['labels']]) #, dataset2['labels'], dataset3['labels'], dataset4['labels'], dataset5['labels']])

data_train = data_train.astype('float') / 255.
labels_train = labels_train
data_test = dataset0['data'].astype('float') / 255.
labels_test = np.array(dataset0['labels'])

n_feat = data_train.shape[1]
n_targets = labels_train.max() + 1


from sknn import mlp

nn = mlp.Classifier(
        layers=[
            mlp.Layer("Tanh", units=n_feat/8),
            mlp.Layer("Sigmoid", units=n_feat/16),
            mlp.Layer("Softmax", units=n_targets)],
        n_iter=50,
        n_stable=10,
        batch_size=25,
        learning_rate=0.002,
        learning_rule="momentum",
        valid_size=0.1,
        verbose=1)

if PRETRAIN:
    from sknn import ae
    ae = ae.AutoEncoder(
            layers=[
                ae.Layer("Tanh", units=n_feat/8),
                ae.Layer("Sigmoid", units=n_feat/16)],
            learning_rate=0.002,
            n_iter=10,
            verbose=1)
    ae.fit(data_train)
    ae.transfer(nn)

nn.fit(data_train, labels_train)


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

expected = labels_test
predicted = nn.predict(data_test)

print("Classification report for classifier %s:\n%s\n" % (
    nn, classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % confusion_matrix(expected, predicted))
