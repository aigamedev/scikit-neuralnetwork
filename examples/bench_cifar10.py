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

net = DBN(
    [n_feat, n_feat / 3, n_targets],
    epochs=50,
    learn_rates=0.03,
    verbose=1,
    )
net.fit(data_train, labels_train)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

expected = labels_test
predicted = net.predict(data_test)

print "Classification report for classifier %s:\n%s\n" % (
    net, classification_report(expected, predicted))
print "Confusion matrix:\n%s" % confusion_matrix(expected, predicted)