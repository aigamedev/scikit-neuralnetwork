import cPickle
import numpy as np

def load(name):
    with open(name, 'rb') as f:
        return cPickle.load(f)

dataset1 = load('data_batch_1')
dataset2 = load('data_batch_2')
dataset3 = load('data_batch_3')

data_train = np.vstack([dataset1['data'], dataset2['data']])
labels_train = np.hstack([dataset1['labels'], dataset2['labels']])

data_train = data_train.astype('float') / 255.
labels_train = labels_train
data_test = dataset3['data'].astype('float') / 255.
labels_test = np.array(dataset3['labels'])

n_feat = data_train.shape[1]
n_targets = labels_train.max() + 1

import sys
import logging
logging.basicConfig(format="%(message)s", level=logging.DEBUG, stream=sys.stdout)

from sknn.mlp import MultiLayerPerceptronClassifier
net = MultiLayerPerceptronClassifier(
    [("Rectifier", n_feat*2/3), ("Rectifier", n_feat*1/3), ("Linear", n_targets)],
    n_iter=50,
    n_stable=10,
    learning_rate=0.005,
    valid_size=0.1,
    verbose=1)
net.fit(data_train, labels_train)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

expected = labels_test
predicted = net.predict(data_test)

print "Classification report for classifier %s:\n%s\n" % (
    net, classification_report(expected, predicted))
print "Confusion matrix:\n%s" % confusion_matrix(expected, predicted)