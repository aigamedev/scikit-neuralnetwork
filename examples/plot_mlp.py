# -*- coding: utf-8 -*-
"""\
Visualizing Parameters in a Modern Neural Network
=================================================
"""
from __future__ import (absolute_import, unicode_literals, print_function)
print(__doc__)

__author__ = 'Alex J. Champandard'

import sys
import time
import logging
import argparse
import itertools

import numpy
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification

# The neural network uses the `sknn` logger to output its information.
import logging
logging.basicConfig(format="%(message)s", level=logging.WARNING, stream=sys.stdout)

from sknn.platform import gpu32
from sknn import mlp


# All possible parameter options that can be plotted, separately or combined.
PARAMETERS = {
    'activation': ['Rectifier', 'Tanh', 'Sigmoid'],
    'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
    'dropout': [None, 0.25, 0.5, 0.75],
    'iterations': [100, 200, 500, 1000],
    'output': ['Softmax', 'Linear', 'Gaussian'],
    'regularize': [None, 'L1', 'L2', 'dropout'],
    'rules': ['sgd', 'momentum', 'nesterov', 'adadelta', 'rmsprop'],
    'units': [16, 64, 128, 256],
}

# Grab command line information from the user.
parser = argparse.ArgumentParser()
parser.add_argument('-p','--params', nargs='+', help='Parameter to visualize.',
                    choices=PARAMETERS.keys(), required=True)
args = parser.parse_args()

# Build a list of lists containing all parameter combinations to be tested.
params = []
for p in sorted(PARAMETERS):
    values = PARAMETERS[p]
    # User requested to test against this parameter?
    if p in args.params:
        params.append(values)
    # Otherwise, use the first item of the list as default.
    else:
        params.append(values[:1])

# Build the classifiers for all possible combinations of parameters.
names = []
classifiers = []
for (activation, alpha, dropout, iterations, output, regularize, rule, units) in itertools.product(*params):
    params = {}
    classifiers.append(mlp.Classifier(
        layers=[mlp.Layer(activation, units=units, **params), mlp.Layer(output)], random_state=1,
        n_iter=iterations, n_stable=iterations, regularize=regularize,
        dropout_rate=dropout, learning_rule=rule, learning_rate=alpha),)

    t = []
    for k, v in zip(sorted(PARAMETERS), [activation, alpha, dropout, iterations, output, regularize, rule, units]):
        if k in args.params:
            t.append(str(v))
    names.append(','.join(t))

# Create randomized datasets for visualizations, on three rows.
seed = int(time.time())
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=0, n_clusters_per_class=1)
rng = numpy.random.RandomState(seed+1)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=seed+2),
            make_circles(noise=0.2, factor=0.5, random_state=seed+3),
            linearly_separable]

# Create the figure containing plots for each of the classifiers.
GRID_RESOLUTION = .02
figure = plt.figure(figsize=(18, 9))
i = 1
for X, y in datasets:
    # Preprocess dataset, split into training and test part.
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    # Prepare coordinates of 2D grid to be visualized.
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, GRID_RESOLUTION),
                            numpy.arange(y_min, y_max, GRID_RESOLUTION))

    # Plot the dataset on its own first.
    cm = plt.cm.get_cmap("PRGn")
    cm_bright = ListedColormap(['#FF00FF', '#00FF00'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # Now iterate over every classifier...
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].

        Z = clf.predict_proba(numpy.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right', fontweight='bold')
        i += 1
        sys.stdout.write('.'); sys.stdout.flush()
    sys.stdout.write('\n')

figure.subplots_adjust(left=.02, right=.98)
plt.show()
