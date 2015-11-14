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

import theano.tensor as T
import lasagne.layers
import lasagne.nonlinearities as nl

from ..base import BaseBackend
from ...nn import Layer, Convolution, ansi


class MultiLayerPerceptronBackend(BaseBackend):
    """
    Abstract base class for wrapping the multi-layer perceptron functionality
    from Lasagne.
    """

    def __init__(self, spec):
        super(MultiLayerPerceptronBackend, self).__init__(spec)
        self.mlp = None
        self.ds = None
        self.vs = None
        self.f = None
        self.trainer = None
        self.cost = None

    def _create_mlp_trainer(self, params):
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

            if self.regularize is None:
                self.regularize = 'dropout'

            log.warning('Dropout not yet fully implemented.')

            """
            # Pass all the parameters to pylearn2 as a custom cost function.
            self.cost = dropout.Dropout(
                default_input_include_prob=default_prob,
                default_input_scale=default_scale,
                input_include_probs=dropout_probs, input_scales=dropout_scales)
             """

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
                raise NotImplementedError
                """
                l1 = mlp_cost.L1WeightDecay(layer_decay)
                self.cost = cost.SumOfCosts([mlp_default_cost,l1])
                """
            else: # Default is 'L2'.
                raise NotImplementedError
                """
                if self.regularize is None:
                    self.regularize = 'L2'

                l2 =  mlp_cost.WeightDecay(layer_decay)
                self.cost = cost.SumOfCosts([mlp_default_cost,l2])
                """

        return self._create_trainer(params, self.cost)

    def _create_trainer(self, params, cost):
        if self.learning_rule != 'sgd':                            
            raise NotImplementedError(
                "Learning rule type `%s` is not supported." % self.learning_rule)

        return lasagne.updates.sgd(cost, params,
                                   learning_rate=self.learning_rate)

    def _create_convolution_layer(self, name, layer, network):
        self._check_layer(layer,
                          required=['channels', 'kernel_shape'],
                          optional=['kernel_stride', 'border_mode', 'pool_shape', 'pool_type'])

        nonlinearities = {'Rectifier': nl.rectify,
                          'Signmoid': nl.sigmoid,
                          'Tanh': nl.tanh,
                          'Softmax': nl.softmax,
                          'Linear': nl.linear}
        assert layer.type in nonlinearities,\
            "Convolution layer type `%s` is not supported." % layer.type
 
        network = lasagne.layers.Conv2DLayer(
                        network,
                        num_filters=layer.channels,
                        filter_size=layer.kernel_shape,
                        nonlinearity=nonlinearities[layer.type])

        if layer.pool_shape != (1, 1):
            network = lasagne.layers.Pool2DLayer(
                        network,
                        pool_size=layer.pool_shape,
                        stride=layer.pool_stride,
                        mode=border_mode)

        return network

    def _create_layer(self, name, layer, network):
        if isinstance(layer, Convolution):
            return self._create_convolution_layer(name, layer, irange)

        nonlinearities = {'Rectifier': nl.rectify,
                          'Signmoid': nl.sigmoid,
                          'Tanh': nl.tanh,
                          'Softmax': nl.softmax,
                          'Linear': nl.linear}

        if layer.dropout:
            network = lasagne.layers.dropout(network, 0.5)

        return lasagne.layers.DenseLayer(network,
                                         num_units=layer.units,
                                         nonlinearity=nonlinearities[layer.type])

    def _create_mlp(self, X):
        self.tensor_input = T.matrix('X')
        self.tensor_output = T.vector('y')
        network = lasagne.layers.InputLayer((None, X.shape[1]), self.tensor_input)

        # Create the layers one by one, connecting to previous.
        self.mlp = []
        for i, layer in enumerate(self.layers):
            
            """
            TODO: Refactor this into common wrapper code.

            fan_in = self.unit_counts[i]
            fan_out = self.unit_counts[i + 1]

            lim = numpy.sqrt(6) / numpy.sqrt(fan_in + fan_out)
            if layer.type == 'Tanh':
                lim *= 1.1 * lim
            elif layer.type in ('Rectifier', 'Maxout'):
                # He, Rang, Zhen and Sun, converted to uniform.
                lim *= numpy.sqrt(2.0)
            elif layer.type == 'Sigmoid':
                lim *= 4.0
            """

            # TODO: self.random_state
            network = self._create_layer(layer.name, layer, network)
            self.mlp.append(network)

        log.info(
            "Initializing neural network with %i layers, %i inputs and %i outputs.",
            len(self.layers), self.unit_counts[0], self.layers[-1].units)

        """
        TODO: Display the network's layers for information.

        for l, p, count in zip(self.layers, self.mlp.layers, self.unit_counts[1:]):
            space = p.get_output_space()
            if isinstance(l, Convolution):                
                log.debug("  - Convl: {}{: <10}{} Output: {}{: <10}{} Channels: {}{}{}".format(
                    ansi.BOLD, l.type, ansi.ENDC,
                    ansi.BOLD, repr(space.shape), ansi.ENDC,
                    ansi.BOLD, space.num_channels, ansi.ENDC))

                # NOTE: Numbers don't match up exactly for pooling; one off. The logic is convoluted!
                # assert count == numpy.product(space.shape) * space.num_channels,\
                #     "Mismatch in the calculated number of convolution layer outputs."
            else:
                log.debug("  - Dense: {}{: <10}{} Units:  {}{: <4}{}".format(
                    ansi.BOLD, l.type, ansi.ENDC, ansi.BOLD, l.units, ansi.ENDC))
                assert count == space.get_total_dimension(),\
                    "Mismatch in the calculated number of dense layer outputs."
        """

        if self.weights is not None:
            l  = min(len(self.weights), len(self.mlp))
            log.info("Reloading parameters for %i layer weights and biases." % (l,))
            self._array_to_mlp(self.weights, self.mlp)
            self.weights = None

        log.debug("")

        output = lasagne.layers.get_output(network, deterministic=True)
        self.f = theano.function([self.tensor_input], output) # allow_input_downcast=True

    def _initialize_impl(self, X, y=None):
        if self.mlp is None:            
            self._create_mlp(X)

        # Can do partial initialization when predicting, no trainer needed.
        if y is None:
            return

        if self.valid_size > 0.0:
            assert self.valid_set is None, "Can't specify valid_size and valid_set together."
            X, X_v, y, y_v = sklearn.cross_validation.train_test_split(
                                X, y,
                                test_size=self.valid_size,
                                random_state=self.random_state)
            self.valid_set = X_v, y_v

        """
        self.ds = self._create_dataset(self.input_space, X, y)
        if self.valid_set is not None:
            X_v, y_v = self.valid_set
            input_space = self._create_input_space(X_v)
            self.vs = self._create_dataset(input_space, X_v, y_v)
        else:
            self.vs = None
        """

        params = lasagne.layers.get_all_params(self.mlp, trainable=True)
        self.trainer = self._create_mlp_trainer(params)
        self.trainer.setup(self.mlp, self.ds)
        return X, y

    def _predict_impl(self, X):
        if not self.is_initialized:
            self._initialize_impl(X)
        return self.f(X)

    def _train_impl(self, X, y):
        if self.is_convolution:
            X = self.ds.view_converter.topo_view_to_design_mat(X)
        self.ds.X, self.ds.y = X, y

        self._train_layer(self.trainer, self.mlp, self.ds)

    @property
    def is_initialized(self):
        """Check if the neural network was setup already.
        """
        return not (self.ds is None or self.f is None)

    def _mlp_to_array(self):
        result = [(l.W.value, l.b.value) for l in self.mlp.layers]
        print(result)
        return result

    def _array_to_mlp(self, array, nn):
        for layer, (weights, biases) in zip(nn, array):
            ws = tuple(layer.W.shape.eval())
            assert ws == weights.shape, "Layer weights shape mismatch: %r != %r" %\
                                        (ws, weights.shape)
            layer.W.set_value(weights)

            bs = tuple(layer.b.shape.eval())
            assert bs == biases.shape, "Layer biases shape mismatch: %r != %r" %\
                                       (bs, biases.shape)
            layer.b.set_value(biases)
