# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

import os
import sys
import time
import logging
import itertools

log = logging.getLogger('sknn')


import numpy

from .pywrap2 import (datasets, space, sgd)
from .pywrap2 import learning_rule as lr, termination_criteria as tc
from .dataset import SparseDesignMatrix, FastVectorSpace


class NeuralNetwork(object):

    def __init__(self, spec):
        self.spec = spec
    
    def __getattr__(self, key):
        return getattr(self.spec, key)

    def _create_input_space(self, X):
        if self.spec.is_convolution:
            # Using `b01c` arrangement of data, see this for details:
            #   http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html
            # input: (batch size, channels, rows, columns)
            # filters: (number of filters, channels, rows, columns)
            return space.Conv2DSpace(shape=X.shape[1:3], num_channels=X.shape[-1])
        else:
            InputSpace = space.VectorSpace if self.spec.debug else FastVectorSpace
            return InputSpace(X.shape[-1])

    def _create_dataset(self, input_space, X, y=None):
        if self.spec.is_convolution:
            view = input_space.get_origin_batch(X.shape[0])
            return datasets.DenseDesignMatrix(topo_view=view, y=y)
        else:
            if all([isinstance(a, numpy.ndarray) for a in (X, y) if a is not None]):
                return datasets.DenseDesignMatrix(X=X, y=y)
            else:
                return SparseDesignMatrix(X=X, y=y)

    def _create_trainer(self, dataset, cost):
        logging.getLogger('pylearn2.monitor').setLevel(logging.WARNING)
        if dataset is not None:
            termination_criterion = tc.MonitorBased(
                channel_name='objective',
                N=self.n_stable-1,
                prop_decrease=self.f_stable)
        else:
            termination_criterion = None

        if self.learning_rule == 'sgd':
            self._learning_rule = None
        elif self.learning_rule == 'adagrad':
            self._learning_rule = lr.AdaGrad()
        elif self.learning_rule == 'adadelta':
            self._learning_rule = lr.AdaDelta()
        elif self.learning_rule == 'momentum':
            self._learning_rule = lr.Momentum(self.learning_momentum)
        elif self.learning_rule == 'nesterov':
            self._learning_rule = lr.Momentum(self.learning_momentum, nesterov_momentum=True)
        elif self.learning_rule == 'rmsprop':
            self._learning_rule = lr.RMSProp()
        else:
            raise NotImplementedError(
                "Learning rule type `%s` is not supported." % self.learning_rule)

        return sgd.SGD(
            cost=cost,
            batch_size=self.batch_size,
            learning_rule=self._learning_rule,
            learning_rate=self.learning_rate,
            termination_criterion=termination_criterion,
            monitoring_dataset=dataset)
