# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, unicode_literals, print_function)

__all__ = ['MultiLayerPerceptronBackend']

import os
import sys
import math
import time
import types
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
from ...nn import Layer, Convolution, Native, ansi


def explin(x):
    return x * (x>=0) + (x<0) * (T.exp(x) - 1)


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
        self.validator = None
        self.regularizer = None

    def _create_mlp_trainer(self, params):
        # Aggregate all regularization parameters into common dictionaries.
        layer_decay = {}
        if self.regularize in ('L1', 'L2') or any(l.weight_decay for l in self.layers):
            wd = self.weight_decay or 0.0001
            for l in self.layers:
                layer_decay[l.name] = l.weight_decay or wd
        assert len(layer_decay) == 0 or self.regularize in ('L1', 'L2', None)

        if len(layer_decay) > 0:
            if self.regularize is None:
                self.auto_enabled['regularize'] = 'L2'
            regularize = self.regularize or 'L2'
            penalty = getattr(lasagne.regularization, regularize.lower())
            apply_regularize = lasagne.regularization.apply_penalty
            self.regularizer = sum(layer_decay[s.name] * apply_regularize(l.get_params(regularizable=True), penalty)
                                   for s, l in zip(self.layers, self.mlp))

        if self.normalize is None and any([l.normalize != None for l in self.layers]):
            self.auto_enabled['normalize'] = 'batch'

        cost_functions = {'mse': 'squared_error', 'mcc': 'categorical_crossentropy'}
        loss_type = self.loss_type or ('mcc' if self.is_classifier else 'mse')
        assert loss_type in cost_functions,\
                    "Loss type `%s` not supported by Lasagne backend." % loss_type
        self.cost_function = getattr(lasagne.objectives, cost_functions[loss_type])
        cost_symbol = self.cost_function(self.trainer_output, self.data_output)
        cost_symbol = lasagne.objectives.aggregate(cost_symbol.T, self.data_mask, mode='mean')

        if self.regularizer is not None:
            cost_symbol = cost_symbol + self.regularizer
        return self._create_trainer_function(params, cost_symbol)

    def _create_trainer_function(self, params, cost):
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

        trainer = theano.function([self.data_input, self.data_output, self.data_mask], cost,
                                   updates=self._learning_rule,
                                   on_unused_input='ignore',
                                   allow_input_downcast=True)

        compare = self.cost_function(self.network_output, self.data_correct).mean()
        validator = theano.function([self.data_input, self.data_correct], compare,
                                    allow_input_downcast=True)
        return trainer, validator

    def _get_activation(self, l):
        nonlinearities = {'Rectifier': nl.rectify,
                          'Sigmoid': nl.sigmoid,
                          'Tanh': nl.tanh,
                          'Softmax': nl.softmax,
                          'Linear': nl.linear,
                          'ExpLin': explin}

        assert l.type in nonlinearities,\
            "Layer type `%s` is not supported for `%s`." % (l.type, l.name)
        return nonlinearities[l.type]

    def _create_convolution_layer(self, name, layer, network):
        self._check_layer(layer,
                          required=['channels', 'kernel_shape'],
                          optional=['units', 'kernel_stride', 'border_mode',
                                    'pool_shape', 'pool_type', 'scale_factor'])

        if layer.scale_factor != (1, 1):
            network = lasagne.layers.Upscale2DLayer(
                            network,
                            scale_factor=layer.scale_factor)

        network = lasagne.layers.Conv2DLayer(
                        network,
                        num_filters=layer.channels,
                        filter_size=layer.kernel_shape,
                        stride=layer.kernel_stride,
                        pad=layer.border_mode,
                        nonlinearity=self._get_activation(layer))

        normalize = layer.normalize or self.normalize
        if normalize == 'batch':
            network = lasagne.layers.batch_norm(network)

        if layer.pool_shape != (1, 1):
            network = lasagne.layers.Pool2DLayer(
                            network,
                            pool_size=layer.pool_shape,
                            stride=layer.pool_shape)

        return network

    def _create_native_layer(self, name, layer, network):
        if layer.units and 'num_units' not in layer.keywords:
            layer.keywords['num_units'] = layer.units
        return layer.type(network, *layer.args, **layer.keywords)

    def _create_layer(self, name, layer, network):
        if isinstance(layer, Native):
            return self._create_native_layer(name, layer, network)

        dropout = layer.dropout or self.dropout_rate
        if dropout is not None:
            network = lasagne.layers.dropout(network, dropout)

        if isinstance(layer, Convolution):
            return self._create_convolution_layer(name, layer, network)

        self._check_layer(layer, required=['units'])
        network = lasagne.layers.DenseLayer(network,
                                            num_units=layer.units,
                                            nonlinearity=self._get_activation(layer))

        normalize = layer.normalize or self.normalize
        if normalize == 'batch':
            network = lasagne.layers.batch_norm(network)
        return network

    def _create_mlp(self, X, w=None):
        self.data_input = T.tensor4('X') if self.is_convolution(input=True) else T.matrix('X')
        self.data_output = T.tensor4('y') if self.is_convolution(output=True) else T.matrix('y')
        self.data_mask = T.vector('m') if w is not None else T.scalar('m')
        self.data_correct = T.matrix('yp')

        lasagne.random.get_rng().seed(self.random_state)

        shape = list(X.shape)
        network = lasagne.layers.InputLayer([None]+shape[1:], self.data_input)

        # Create the layers one by one, connecting to previous.
        self.mlp = []
        for i, layer in enumerate(self.layers):
            network = self._create_layer(layer.name, layer, network)
            network.name = layer.name
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
            elif isinstance(l, Native):
                log.debug("  - Nativ: {}{: <10}{} Output: {}{: <10}{} Channels: {}{}{}".format(
                    ansi.BOLD, l.type.__name__, ansi.ENDC,
                    ansi.BOLD, repr(space[2:]), ansi.ENDC,
                    ansi.BOLD, space[1], ansi.ENDC))
            else:
                log.debug("  - Dense: {}{: <10}{} Units:  {}{: <4}{}".format(
                    ansi.BOLD, l.type, ansi.ENDC, ansi.BOLD, l.units, ansi.ENDC))
                assert count == space[1],\
                    "Mismatch in the calculated number of dense layer outputs. {} != {}".format(count, space[1])

        if self.weights is not None:
            l  = min(len(self.weights), len(self.mlp))
            log.info("Reloading parameters for %i layer weights and biases." % (l,))
            self._array_to_mlp(self.weights, self.mlp)
            self.weights = None

        log.debug("")

        self.network_output = lasagne.layers.get_output(network, deterministic=True)
        self.trainer_output = lasagne.layers.get_output(network, deterministic=False)
        self.f = theano.function([self.data_input], self.network_output, allow_input_downcast=True)

    def _conv_transpose(self, arr):
        ok = arr.shape[-1] not in (1,3) and arr.shape[1] in (1,3)
        return arr if ok else numpy.transpose(arr, (0, 3, 1, 2))

    def _initialize_impl(self, X, y=None, w=None):
        if self.is_convolution(input=True):
            X = self._conv_transpose(X)
        if y is not None and self.is_convolution(output=True):
            y = self._conv_transpose(y)

        if self.mlp is None:
            self._create_mlp(X, w)

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

        if self.valid_set and self.is_convolution():
            X_v, y_v = self.valid_set
            if X_v.shape[-2:] != X.shape[-2:]:
                self.valid_set = numpy.transpose(X_v, (0, 3, 1, 2)), y_v

        params = []
        for spec, mlp_layer in zip(self.layers, self.mlp):
            if spec.frozen: continue
            params.extend(mlp_layer.get_params())

        self.trainer, self.validator = self._create_mlp_trainer(params)
        return X, y

    def _predict_impl(self, X):
        if self.is_convolution():
            X = numpy.transpose(X, (0, 3, 1, 2))

        y = None
        for Xb, _, _, idx  in self._iterate_data(self.batch_size, X, y, shuffle=False):
            yb = self.f(Xb)
            if y is None:
                if X.shape[0] <= self.batch_size:
                    y = yb
                    break
                else:
                    y = numpy.zeros(X.shape[:1] + yb.shape[1:], dtype=theano.config.floatX)
            y[idx] = yb
        return y

    def _iterate_data(self, batch_size, X, y=None, w=None, shuffle=False):
        def cast(array, indices):
            if array is None:
                return None

            # Support for pandas.DataFrame, requires custom indexing.
            if type(array).__name__ == 'DataFrame':
                array = array.loc[indices]
            else:
                array = array[indices]

                # Support for scipy.sparse; convert after slicing.
                if hasattr(array, 'todense'):
                    array = array.todense()

            return array.astype(theano.config.floatX)

        total_size = X.shape[0]
        indices = numpy.arange(total_size)
        if shuffle:
            numpy.random.shuffle(indices)

        for index in range(0, total_size, batch_size):
            excerpt = indices[index:index + batch_size]
            Xb, yb, wb = cast(X, excerpt), cast(y, excerpt), cast(w, excerpt)
            yield Xb, yb, wb, excerpt

    def _print(self, text):
        if self.verbose:
            sys.stdout.write(text)
            sys.stdout.flush()

    def _batch_impl(self, X, y, w, processor, mode, output, shuffle):
        progress, batches = 0, X.shape[0] / self.batch_size
        loss, count = 0.0, 0
        for Xb, yb, wb, _ in self._iterate_data(self.batch_size, X, y, w, shuffle):
            self._do_callback('on_batch_start', locals())

            if mode == 'train':
                loss += processor(Xb, yb, wb if wb is not None else 1.0)
            else:
                loss += processor(Xb, yb)
            count += 1

            while count / batches > progress / 60:
                self._print(output)
                progress += 1

            self._do_callback('on_batch_finish', locals())

        self._print('\r')
        return loss / count

    def _train_impl(self, X, y, w=None):
        return self._batch_impl(X, y, w, self.trainer, mode='train', output='.', shuffle=True)

    def _valid_impl(self, X, y, w=None):
        return self._batch_impl(X, y, w, self.validator, mode='valid', output=' ', shuffle=False)

    @property
    def is_initialized(self):
        """Check if the neural network was setup already.
        """
        return not (self.f is None)

    def _mlp_get_layer_params(self, layer):
        """Traverse the Lasagne network accumulating parameters until
        reaching the next "major" layer specified and named by the user.
        """
        assert layer.name is not None, "Expecting this layer to have a name."

        params = []
        while hasattr(layer, 'input_layer'):
            params.extend(layer.get_params())
            layer = layer.input_layer
            if layer.name is not None:
                break
        return params

    def _mlp_to_array(self):
        return [[p.get_value() for p in self._mlp_get_layer_params(l)] for l in self.mlp]

    def _array_to_mlp(self, array, nn):
        for layer, data in zip(nn, array):
            if data is None:
                continue

            # Handle namedtuple format returned by get_parameters() as special case.
            # Must remove the last `name` item in the tuple since it's not a parameter.
            string_types = getattr(types, 'StringTypes', tuple([str]))
            data = tuple([d for d in data if not isinstance(d, string_types)])

            params = self._mlp_get_layer_params(layer)
            assert len(data) == len(params),\
                            "Mismatch in data size for layer `%s`. %i != %i"\
                            % (layer.name, len(data), len(params))

            for p, d in zip(params, data):
                ps = tuple(p.shape.eval())
                assert ps == d.shape, "Layer parameter shape mismatch: %r != %r" % (ps, d.shape)
                p.set_value(d.astype(theano.config.floatX))
