from __future__ import (absolute_import, unicode_literals, print_function)

__author__ = 'ssamot, alexjc'


import numpy as np

from .nn import NeuralNetwork


class IncrementalMinMaxScaler():

    def __init__(self, feature_range=(-1.0, 1.0)):
        self.feature_range = feature_range
        self.changed = False
        self.init = False
        self.times = 0

    def fit(self, X, y=None):
        self.changed = False
        self.times += 1
        if (not self.init):
            self.min_ = np.array(X[0], dtype=np.float64)
            self.max_ = np.array(X[0], dtype=np.float64)
            self.data_min = self.min_
            self.data_max = self.max_
            self.init = True
        else:
            X = np.array(X, ndmin=2)
            X = np.append(X, [self.data_min], axis=0)
            X = np.append(X, [self.data_max], axis=0)

        feature_range = self.feature_range
        data_min = np.min(X, axis=0)
        data_max = np.max(X, axis=0)

        if not (self.data_min == data_min).all():
            self.changed = True

        if not (self.data_max == data_max).all():
            self.changed = True

        self.data_min = data_min
        self.data_max = data_max

        data_range = data_max - data_min

        data_range[data_range == 0.0] = 1.0
        data_range[data_range == 0] = 1.0

        self.scale_ = (feature_range[1] - feature_range[0]) / data_range
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_range = data_range
        self.data_min = data_min
        return self

    def transform(self, X):
        assert (len(X.shape) == 2), X
        transformed = (X * self.scale_) + self.min_
        return transformed

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            Input data that will be transformed.
        """
        reverse_transformed = (X - self.min_) / self.scale_
        return reverse_transformed
