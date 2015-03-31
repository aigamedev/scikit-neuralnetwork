from __future__ import (absolute_import, unicode_literals)

import logging

import numpy as np
import theano

from pylearn2.datasets import DenseDesignMatrix
from pylearn2.training_algorithms import sgd, bgd
from pylearn2.models import mlp, maxout
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms.learning_rule import AdaGrad, RMSProp, Momentum


log = logging.getLogger('sknn')


class NeuralNetwork(object):

    """
    SK-learn like interface for pylearn2
    Notice how training the model and the training algorithm are now part of the same class, which I actually quite like
    This class is focused a bit on online learning, so you might need to modify it to include other pylearn2 options if
    you have access all your data upfront
    """

    def __init__(
            self,
            layers,
            dropout=False,
            learning_rate=0.001):
        """
        :param layers: List of tuples of types of layers alongside the number of neurons
        :param learning_rate: The learning rate for all layers
        :return:
        """

        self.layers = layers

        self.ds = None
        self.f = None

        if dropout:
            self.cost = "Dropout"
            self.weight_scale = None
        else:
            self.cost = None
            self.weight_scale = None

        self.learning_rate = learning_rate
        #self.learning_rule = Momentum(0.9)

        self.learning_rule = Momentum(0.9)
        #self.learning_rule = None
        #self.learning_rule = RMSProp()

    def create_trainer(self):
        return sgd.SGD(
            learning_rate=self.learning_rate,
            cost=self.cost,
            batch_size=1,
            learning_rule=self.learning_rule)

    def initialize(self, X, y):
        log.info(
            "Initializing neural network with %i layers.",
            len(self.layers))

        pylearn2mlp_layers = []
        self.units_per_layer = []
        # input layer units
        self.units_per_layer += [X.shape[1]]

        for layer in self.layers[:-1]:
            self.units_per_layer += [layer[1]]

        # Output layer units
        self.units_per_layer += [y.shape[1]]

        log.debug("Units per layer: %r.", self.units_per_layer)

        for i, layer in enumerate(self.layers[:-1]):

            fan_in = self.units_per_layer[i] + 1
            fan_out = self.units_per_layer[i + 1]
            lim = np.sqrt(6) / (np.sqrt(fan_in + fan_out))
            layer_name = "Hidden_%i_%s" % (i, layer[0])
            activate_type = layer[0]
            if i == 0:
                first_hidden_name = layer_name
            if activate_type == "RectifiedLinear":
                hidden_layer = mlp.RectifiedLinear(
                    dim=layer[1],
                    layer_name=layer_name,
                    irange=lim,
                    W_lr_scale=self.weight_scale)
            elif activate_type == "Sigmoid":
                hidden_layer = mlp.Sigmoid(
                    dim=layer[1],
                    layer_name=layer_name,
                    irange=lim,
                    W_lr_scale=self.weight_scale)
            elif activate_type == "Tanh":
                hidden_layer = mlp.Tanh(
                    dim=layer[1],
                    layer_name=layer_name,
                    irange=lim,
                    W_lr_scale=self.weight_scale)
            elif activate_type == "Maxout":
                hidden_layer = maxout.Maxout(
                    num_units=layer[1],
                    num_pieces=layer[2],
                    layer_name=layer_name,
                    irange=lim,
                    W_lr_scale=self.weight_scale)

            else:
                raise NotImplementedError(
                    "Layer of type %s are not implemented yet" %
                    layer[0])
            pylearn2mlp_layers += [hidden_layer]

        output_layer_info = self.layers[-1]
        output_layer_name = "Output_%s" % output_layer_info[0]

        # fan_in = self.units_per_layer[-2] + 1
        # fan_out = self.units_per_layer[-1]
        # lim = np.sqrt(6) / (np.sqrt(fan_in + fan_out))

        if output_layer_info[0] == "Linear":
            output_layer = mlp.Linear(
                dim=self.units_per_layer[-1],
                layer_name=output_layer_name,
                irange=0.00001,
                W_lr_scale=self.weight_scale)

        if self.cost is not None:
            if self.cost == "Dropout":
                self.cost = Dropout(
                    input_include_probs={first_hidden_name: 1.0},
                    input_scales={first_hidden_name: 1.})

        if output_layer_info[0] == "LinearGaussian":
            output_layer = mlp.LinearGaussian(
                init_beta=0.1,
                min_beta=0.001,
                max_beta=1000,
                beta_lr_scale=None,

                dim=self.units_per_layer[-1],
                layer_name=output_layer_name,
                irange=0.1,

                W_lr_scale=self.weight_scale)

        pylearn2mlp_layers += [output_layer]

        self.mlp = mlp.MLP(pylearn2mlp_layers, nvis=self.units_per_layer[0])
        self.ds = DenseDesignMatrix(X=X, y=y)
        self.trainer = self.create_trainer()
        self.trainer.setup(self.mlp, self.ds)
        inputs = self.mlp.get_input_space().make_theano_batch()
        self.f = theano.function([inputs], self.mlp.fprop(inputs))

    def fit(self, X, y):
        """
        :param X: Training data
        :param y:
        :return:
        """

        if self.ds is None:
            self.initialize(X, y)

        if self.trainer is None:
            self.trainer = self.create_trainer()
            self.trainer.setup(self.mlp, self.ds)

        ds = self.ds
        ds.X, ds.y = X, y
        self.trainer.train(dataset=ds)

        return self

    def predict(self, X, n_out=None):
        """

        :param X:
        :return:
        """

        if self.ds is None:
            assert n_out is not None,\
                "Call initialize() first or specify number of outputs."

            self.initialize(X, np.array([np.zeros(n_out)]))

        return self.f(X)

    def __getstate__(self):

        self.ds.X = self.ds.X[0:2]
        self.ds.y = self.ds.y[0:2]

        d = dict(self.__dict__)
        del d['f']
        del d['trainer']

        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

        self.trainer = None

        inputs = self.mlp.get_input_space().make_theano_batch()
        self.f = theano.function([inputs], self.mlp.fprop(inputs))
