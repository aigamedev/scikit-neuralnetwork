# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

from sklearn import datasets, cross_validation
from sknn.mlp import Classifier, Layer, Convolution


# Load the data and split it into subsets for training and testing.
digits = datasets.load_digits()
X = digits.images
y = digits.target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


# Create a neural network that uses convolution to scan the input images.
nn = Classifier(
    layers=[
        Convolution('Rectifier', channels=12, kernel_shape=(3, 3), border_mode='full'),
        Convolution('Rectifier', channels=8, kernel_shape=(3, 3), border_mode='valid'),
        Layer('Rectifier', units=64),
        Layer('Softmax')],
    learning_rate=0.002,
    valid_size=0.2,
    n_stable=10,
    verbose=True)

nn.fit(X_train, y_train)


# Determine how well it does on training data and unseen test data.
print('\nTRAIN SCORE', nn.score(X_train, y_train))
print('TEST SCORE', nn.score(X_test, y_test))

y_pred = nn.predict(X_test)


# Show some training images and some test images too.
import matplotlib.pyplot as pylab

for index, (image, label) in enumerate(zip(digits.images[:6], digits.target[:6])):
    pylab.subplot(2, 6, index + 1)
    pylab.axis('off')
    pylab.imshow(image, cmap=pylab.cm.gray_r, interpolation='nearest')
    pylab.title('Training: %i' % label)

for index, (image, prediction) in enumerate(zip(X_test[:6], y_pred[:6])):
    pylab.subplot(2, 6, index + 7)
    pylab.axis('off')
    pylab.imshow(image.reshape((8,8)), cmap=pylab.cm.gray_r, interpolation='nearest')
    pylab.title('Predicts: %i' % prediction)

pylab.show()
