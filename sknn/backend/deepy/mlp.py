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
from deepy.layers import Dense, Softmax, Dropout
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
        super(MultiLayerPerceptron, self).__init__(spec)
        self.iterations = 0        
        self.trainer = None
        self.mlp = None

    @property
    def is_convolution(self):
        return False

    def _create_mlp_trainer(self, dataset):
        # Aggregate all the dropout parameters into shared dictionaries.
        dropout_probs, dropout_scales = {}, {}
        for l in [l for l in self.layers if l.dropout is not None]:
            incl = 1.0 - l.dropout
            dropout_probs[l.name] = incl
            dropout_scales[l.name] = 1.0 / incl
        assert len(dropout_probs) == 0 or self.regularize in ('dropout', None)

        if self.regularize == 'dropout' or len(dropout_probs) > 0:
            # Use the globally specified dropout rate when there are no layer-specific ones.
            incl = 1.0 - (self.dropout_rate or 0.5)
            default_prob, default_scale = incl, 1.0 / incl
            self.regularize = 'dropout'
            
            # Pass all the parameters to pylearn2 as a custom cost function.
            self.cost = dropout.Dropout(
                default_input_include_prob=default_prob,
                default_input_scale=default_scale,
                input_include_probs=dropout_probs, input_scales=dropout_scales)

        # Aggregate all regularization parameters into common dictionaries.
        layer_decay = {}
        if self.regularize in ('L1', 'L2') or any(l.weight_decay for l in self.layers):
            wd = self.weight_decay or 0.0001
            for l in self.layers:
                layer_decay[l.name] = l.weight_decay or wd
        assert len(layer_decay) == 0 or self.regularize in ('L1', 'L2', None)

        if len(layer_decay) > 0:
            mlp_default_cost = self.mlp.get_default_cost()
            if self.regularize == 'L1':
                l1 = mlp_cost.L1WeightDecay(layer_decay)
                self.cost = cost.SumOfCosts([mlp_default_cost,l1])
            else: # Default is 'L2'.
                self.regularize = 'L2'
                l2 =  mlp_cost.WeightDecay(layer_decay)
                self.cost = cost.SumOfCosts([mlp_default_cost,l2])

        return self._create_trainer(dataset, self.cost)

    def _create_mlp(self):
        model = NeuralRegressor(input_dim=self.unit_counts[0])
        initializer = UniformInitializer(seed=self.random_state)

        for l, n in zip(self.layers, self.unit_counts[1:]):
            t = None
            if l.type in ('Tanh', 'Sigmoid'): t = l.type.lower()
            if l.type in ('Rectifier', 'Maxout'): t = 'relu'
            if l.type in ('Linear', 'Softmax'): t = 'linear'
            assert t is not None, "Unknown activation type `%s`." % l.type

            model.stack_layer(Dense(n, t, init=initializer))
            if l.type == 'Softmax':
                model.stack_layer(Softmax())

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

    def _reshape(self, X, y=None):
        # TODO: Common for all backends.
        if y is not None and y.ndim == 1:
            y = y.reshape((y.shape[0], 1))
        if self.is_convolution and X.ndim == 3:
            X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
        if self.is_convolution and X.ndim == 2:
            size = math.sqrt(X.shape[1])
            assert size.is_integer(),\
                "Input array is not in image shape, and could not assume a square."
            X = X.reshape((X.shape[0], int(size), int(size), 1))
        if not self.is_convolution and X.ndim > 2:
            X = X.reshape((X.shape[0], numpy.product(X.shape[1:])))
        return X, y

    def _train_impl(self, X, y):
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
