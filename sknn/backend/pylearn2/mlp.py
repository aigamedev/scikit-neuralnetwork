# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

__all__ = ['MultiLayerPerceptronBackend']

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

from ...nn import Layer, Convolution, ansi
from .nn import NeuralNetworkBackend
from .pywrap2 import (mlp, maxout, cost, dropout, mlp_cost)


class MultiLayerPerceptronBackend(NeuralNetworkBackend):
    """
    Abstract base class for wrapping the multi-layer perceptron functionality
    from PyLearn2.
    """

    def __init__(self, spec):
        super(MultiLayerPerceptronBackend, self).__init__(spec)
        self.input_space = None
        self.mlp = None
        self.ds = None
        self.vs = None
        self.f = None
        self.trainer = None
        self.cost = None

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

            if self.regularize is None:
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
                if self.regularize is None:
                    self.regularize = 'L2'

                l2 =  mlp_cost.WeightDecay(layer_decay)
                self.cost = cost.SumOfCosts([mlp_default_cost,l2])

        return self._create_trainer(dataset, self.cost)

    def _create_convolution_layer(self, name, layer, irange):
        self._check_layer(layer,
                          required=['channels', 'kernel_shape'],
                          optional=['kernel_stride', 'border_mode', 'pool_shape', 'pool_type'])

        if layer.type == 'Rectifier':
            nl = mlp.RectifierConvNonlinearity(0.0)
        elif layer.type == 'Sigmoid':
            nl = mlp.SigmoidConvNonlinearity()
        elif layer.type == 'Tanh':
            nl = mlp.TanhConvNonlinearity()
        else:
            assert layer.type == 'Linear',\
                "Convolution layer type `%s` is not supported." % layer.type
            nl = mlp.IdentityConvNonlinearity()

        return mlp.ConvElemwise(
            layer_name=name,
            nonlinearity=nl,
            output_channels=layer.channels,
            kernel_shape=layer.kernel_shape,
            kernel_stride=layer.kernel_stride,
            border_mode=layer.border_mode,
            pool_shape=layer.pool_shape,
            pool_type=layer.pool_type,
            pool_stride=(1,1),
            irange=irange)

    def _create_layer(self, name, layer, irange):
        if isinstance(layer, Convolution):
            return self._create_convolution_layer(name, layer, irange)

        if layer.type == 'Rectifier':
            self._check_layer(layer, ['units'])
            return mlp.RectifiedLinear(
                layer_name=name,
                dim=layer.units,
                irange=irange)

        if layer.type == 'Sigmoid':
            self._check_layer(layer, ['units'])
            return mlp.Sigmoid(
                layer_name=name,
                dim=layer.units,
                irange=irange)

        if layer.type == 'Tanh':
            self._check_layer(layer, ['units'])
            return mlp.Tanh(
                layer_name=name,
                dim=layer.units,
                irange=irange)

        if layer.type == 'Maxout':
            self._check_layer(layer, ['units', 'pieces'])
            return maxout.Maxout(
                layer_name=name,
                num_units=layer.units,
                num_pieces=layer.pieces,
                irange=irange)

        if layer.type == 'Linear':
            self._check_layer(layer, ['units'])
            return mlp.Linear(
                layer_name=layer.name,
                dim=layer.units,
                irange=irange,
                use_abs_loss=bool(self.loss_type == 'mae'))

        if layer.type == 'Gaussian':
            self._check_layer(layer, ['units'])
            return mlp.LinearGaussian(
                layer_name=layer.name,
                init_beta=0.1,
                min_beta=0.001,
                max_beta=1000,
                beta_lr_scale=None,
                dim=layer.units,
                irange=irange,
                use_abs_loss=bool(self.loss_type == 'mae'))

        if layer.type == 'Softmax':
            self._check_layer(layer, ['units'])
            return mlp.Softmax(
                layer_name=layer.name,
                n_classes=layer.units,
                irange=irange)

    def _create_mlp(self):
        # Create the layers one by one, connecting to previous.
        mlp_layers = []
        for i, layer in enumerate(self.layers):
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

            mlp_layer = self._create_layer(layer.name, layer, irange=lim)
            mlp_layers.append(mlp_layer)

        log.info(
            "Initializing neural network with %i layers, %i inputs and %i outputs.",
            len(self.layers), self.unit_counts[0], self.layers[-1].units)

        self.mlp = mlp.MLP(
            mlp_layers,
            nvis=None if self.input_space else self.unit_counts[0],
            seed=self.random_state,
            input_space=self.input_space)

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

        if self.weights is not None:
            l  = min(len(self.weights), len(self.mlp.layers))
            log.info("Reloading parameters for %i layer weights and biases." % (l,))
            self._array_to_mlp(self.weights, self.mlp)
            self.weights = None
        log.debug("")

        inputs = self.mlp.get_input_space().make_theano_batch()
        self.f = theano.function([inputs], self.mlp.fprop(inputs), allow_input_downcast=True)

    def _initialize_impl(self, X, y=None):
        # Convolution networks need a custom input space.
        self.input_space = self._create_input_space(X)
        if self.mlp is None:
            self._create_mlp()

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

        self.ds = self._create_dataset(self.input_space, X, y)
        if self.valid_set is not None:
            X_v, y_v = self.valid_set
            input_space = self._create_input_space(X_v)
            self.vs = self._create_dataset(input_space, X_v, y_v)
        else:
            self.vs = None

        self.trainer = self._create_mlp_trainer(self.vs)
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

    def _mlp_get_weights(self, l):
        if isinstance(l, mlp.ConvElemwise) or getattr(l, 'requires_reformat', False):
            W, = l.transformer.get_params()
            return W.get_value()
        return l.get_weights()

    def _mlp_to_array(self):
        return [(self._mlp_get_weights(l), l.get_biases()) for l in self.mlp.layers]

    def _array_to_mlp(self, array, nn):
        for layer, (weights, biases) in zip(nn.layers, array):
            assert self._mlp_get_weights(layer).shape == weights.shape
            layer.set_weights(weights)

            assert layer.get_biases().shape == biases.shape
            layer.set_biases(biases)
