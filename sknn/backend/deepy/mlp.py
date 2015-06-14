# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

__all__ = ['Regressor', 'Classifier', 'Layer', 'Convolution']

import os
import sys
import math
import time
import logging
import itertools

log = logging.getLogger('sknn')


import numpy
import theano
import sklearn.base
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.cross_validation

from deepy.dataset import MiniBatches, SequentialDataset
from deepy.networks import NeuralRegressor
from deepy import layers
from deepy.trainers import MomentumTrainer, LearningRateAnnealer
from deepy.utils import UniformInitializer

from ...nn import Layer, Convolution, ansi
from ..base import BaseBackend


class MultiLayerPerceptronBackend(BaseBackend):
    """
    Abstract base class for wrapping the multi-layer perceptron functionality
    from ``deepy``.
    """

    def __init__(self, spec):
        super(MultiLayerPerceptronBackend, self).__init__(spec)
        self.iterations = 0
        self.trainer = None
        self.mlp = None

        l = logging.getLogger('deepy')
        l.setLevel(logging.WARNING)

    @property
    def is_convolution(self):
        return False

    def _create_mlp(self):
        model = NeuralRegressor(input_dim=self.unit_counts[0])
        initializer = UniformInitializer(seed=self.random_state)

        if self.spec.is_convolution:
            model.stack_layer(layers.DimShuffle((0, 'x', 1, 2)))

        for l, n in zip(self.layers, self.unit_counts[1:]):
            t = None
            if l.type in ('Tanh', 'Sigmoid'): t = l.type.lower()
            if l.type in ('Rectifier'): t = 'relu'
            if l.type in ('Linear', 'Softmax'): t = 'linear'
            assert t is not None, "Unknown activation type `%s`." % l.type

            if isinstance(l, Layer):
                # self._check_layer(l, ['units'])
                model.stack_layer(layers.Dense(n, t, init=initializer))
                if l.type == 'Softmax':
                    model.stack_layer(layers.Softmax())
            if isinstance(l, layers.Convolution):
                # self._check_layer(l, ['channel', 'kernel_shape'])
                model.stack_layer(layers.Convolution(
                    activation=t,
                    filter_shape=(l.channels, l.kernel_shape[0], l.kernel_shape[1]),
                    pool_size=l.pool_shape,
                    border_mode=l.border_mode,
                    init=initializer))

        self.mlp = model

    def _initialize_impl(self, X, y=None):
        assert not self.is_initialized,\
            "This neural network has already been initialized."
        self._create_specs(X, y)

        self._create_mlp()
        if y is None:
            return

        if self.valid_size > 0.0:
            assert self.valid_set is None, "Can't specify valid_size and valid_set together."
            X, X_v, y, y_v = sklearn.cross_validation.train_test_split(
                                X, y,
                                test_size=self.valid_size,
                                random_state=self.random_state)
            self.valid_set = X_v, y_v
        self.train_set = X, y
        
        self.trainer = MomentumTrainer(self.mlp)
        self.controllers = [
            self,
            LearningRateAnnealer(self.trainer, patience=self.n_stable, anneal_times=0)]

    def invoke(self):
        """Controller interface for deepy's trainer.
        """
        self.iterations += 1
        return bool(self.iterations >= self.n_iter)

    @property
    def is_initialized(self):
        """Check if the neural network was setup already.
        """
        return self.trainer is not None

    def _train_impl(self, X, y):
        if self.spec.is_convolution:
            X = X.reshape(X.shape[:3])
        self.iterations = 0
        data = zip(X, y)
        self.dataset = SequentialDataset(data)
        minibatches = MiniBatches(self.dataset, batch_size=20)
        self.trainer.run(minibatches, controllers=self.controllers)
        return self

    def _predict_impl(self, X):
        return self.mlp.compute(X)

    def _mlp_to_array(self):
        return []

    def _array_to_mlp(self, array):
        pass
