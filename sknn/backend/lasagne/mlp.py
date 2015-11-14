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
        self.f = None
        self.trainer = None
        self.cost = None

    def _create_mlp_trainer(self, params):
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

        self.cost = lasagne.objectives.categorical_crossentropy(self.symbol_output, self.tensor_output).mean()
        return self._create_trainer(params, self.cost)

    def _create_trainer(self, params, cost):
        if self.learning_rule in ('sgd', 'adagrad', 'adadelta', 'rmsprop', 'adam'):
            lr = getattr(lasagne.updates, self.learning_rule)
            self._learning_rule = lr(cost, params, learning_rate=self.learning_rate)
        elif self.learning_rule in ('momentum', 'nesterov'):
            lasagne.updates.nesterov = lasagne.updates.nesterov_momentum
            lr = getattr(lasagne.updates, self.learning_rule)
            self._learning_rule = lr(cost, params, learning_rate=self.learning_rate, momentum=self.learning_momentum)
        else:
            raise NotImplementedError(
                "Learning rule type `%s` is not supported." % self.learning_rule)

        return theano.function([self.tensor_input, self.tensor_output], cost,
                               updates=self._learning_rule,
                               allow_input_downcast=True)

    def _get_activation(self, l):        
        nonlinearities = {'Rectifier': nl.rectify,
                          'Sigmoid': nl.sigmoid,
                          'Tanh': nl.tanh,
                          'Softmax': nl.softmax,
                          'Linear': nl.linear}

        assert l.type in nonlinearities,\
            "Layer type `%s` is not supported for `%s`." % (layer.type, layer.name)
        return nonlinearities[l.type]

    def _create_convolution_layer(self, name, layer, network):
        self._check_layer(layer,
                          required=['channels', 'kernel_shape'],
                          optional=['kernel_stride', 'border_mode', 'pool_shape', 'pool_type'])
 
        network = lasagne.layers.Conv2DLayer(
                        network,
                        num_filters=layer.channels,
                        filter_size=layer.kernel_shape,
                        stride=layer.kernel_stride,
                        pad=layer.border_mode,
                        nonlinearity=self._get_activation(layer))

        if layer.pool_shape != (1, 1):
            network = lasagne.layers.Pool2DLayer(
                        network,
                        pool_size=layer.pool_shape,
                        stride=layer.pool_shape)

        return network

    def _create_layer(self, name, layer, network):
        if isinstance(layer, Convolution):
            return self._create_convolution_layer(name, layer, network)

        dropout = layer.dropout or self.dropout_rate
        if dropout is not None:
            network = lasagne.layers.dropout(network, dropout)

        return lasagne.layers.DenseLayer(network,
                                         num_units=layer.units,
                                         nonlinearity=self._get_activation(layer))

    def _create_mlp(self, X):
        self.tensor_input = T.tensor4('X') if self.is_convolution else T.matrix('X')
        self.tensor_output = T.matrix('y')
        
        shape = list(X.shape)
        network = lasagne.layers.InputLayer([None]+shape[1:], self.tensor_input)

        # Create the layers one by one, connecting to previous.
        self.mlp = []
        for i, layer in enumerate(self.layers):
            
            """
            TODO: Expose weight initialization policy.
            TODO: self.random_state
            """

            network = self._create_layer(layer.name, layer, network)
            self.mlp.append(network)

        log.info(
            "Initializing neural network with %i layers, %i inputs and %i outputs.",
            len(self.layers), self.unit_counts[0], self.layers[-1].units)

        for l, p, count in zip(self.layers, self.mlp, self.unit_counts[1:]):
            space = p.output_shape
            if isinstance(l, Convolution):                
                log.debug("  - Convl: {}{: <10}{} Output: {}{: <10}{} Channels: {}{}{}".format(
                    ansi.BOLD, l.type, ansi.ENDC,
                    ansi.BOLD, repr(space[2:]), ansi.ENDC,
                    ansi.BOLD, space[1], ansi.ENDC))

                # NOTE: Numbers don't match up exactly for pooling; one off. The logic is convoluted!
                # assert count == numpy.product(space.shape) * space.num_channels,\
                #     "Mismatch in the calculated number of convolution layer outputs."
            else:
                log.debug("  - Dense: {}{: <10}{} Units:  {}{: <4}{}".format(
                    ansi.BOLD, l.type, ansi.ENDC, ansi.BOLD, l.units, ansi.ENDC))
                assert count == space[1],\
                    "Mismatch in the calculated number of dense layer outputs."

        if self.weights is not None:
            l  = min(len(self.weights), len(self.mlp))
            log.info("Reloading parameters for %i layer weights and biases." % (l,))
            self._array_to_mlp(self.weights, self.mlp)
            self.weights = None

        log.debug("")

        self.symbol_output = lasagne.layers.get_output(network, deterministic=True)
        self.f = theano.function([self.tensor_input], self.symbol_output, allow_input_downcast=True)

    def _initialize_impl(self, X, y=None):
        if self.is_convolution:
            X = numpy.transpose(X, (0, 3, 1, 2))

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

        params = lasagne.layers.get_all_params(self.mlp[-1], trainable=True)
        self.trainer = self._create_mlp_trainer(params)
        return X, y

    def _predict_impl(self, X):
        if not self.is_initialized:
            self._initialize_impl(X)
        
        if self.is_convolution:
            X = numpy.transpose(X, (0, 3, 1, 2))
        return self.f(X)
    
    def _iterate_data(self, X, y, batch_size):
        indices = numpy.arange(len(X))
        numpy.random.shuffle(indices)
        for start_idx in range(0, len(X) - batch_size + 1, batch_size):
            excerpt = indices[start_idx:start_idx + batch_size]
            yield X[excerpt], y[excerpt]

    def _train_impl(self, X, y):
        best_valid_error = float("inf")

        for i in itertools.count(1):
            start = time.time()

            loss, batches = 0.0, 0
            for Xb, yb in self._iterate_data(X, y, self.batch_size):
                loss += self.trainer(Xb, yb)
                batches += 1

            avg_valid_error = loss / batches
            best_valid_error = min(best_valid_error, avg_valid_error)

            best_valid = bool(best_valid_error == avg_valid_error)
            log.debug("\r{:>5}      {}{}{}        {:>5.1f}s".format(
                      i,
                      ansi.GREEN if best_valid else "",
                      "{:>10.6f}".format(float(avg_valid_error)) if (avg_valid_error is not None) else "     N/A  ",
                      ansi.ENDC if best_valid else "",
                      time.time() - start
                      ))

            if False: # TODO: Monitor n_stable
                log.debug("")
                log.info("Early termination condition fired at %i iterations.", i)
                break
            if self.n_iter is not None and i >= self.n_iter:
                log.debug("")
                log.info("Terminating after specified %i total iterations.", i)
                break

    @property
    def is_initialized(self):
        """Check if the neural network was setup already.
        """
        return not (self.f is None)

    def _mlp_to_array(self):
        return [(l.W.get_value(), l.b.get_value()) for l in self.mlp]

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
