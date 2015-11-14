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
from .dataset import DenseDesignMatrix, SparseDesignMatrix, FastVectorSpace

from ...nn import ansi
from ..base import BaseBackend


class NeuralNetworkBackend(BaseBackend):

    def _create_input_space(self, X):
        if self.is_convolution:
            # Using `b01c` arrangement of data, see this for details:
            #   http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html
            # input: (batch size, channels, rows, columns)
            # filters: (number of filters, channels, rows, columns)
            return space.Conv2DSpace(shape=X.shape[1:3], num_channels=X.shape[-1])
        else:
            InputSpace = space.VectorSpace if self.debug else FastVectorSpace
            return InputSpace(X.shape[-1])

    def _create_dataset(self, input_space, X, y=None):
        if self.is_convolution:
            view = input_space.get_origin_batch(X.shape[0])
            return DenseDesignMatrix(topo_view=view, y=y, mutator=self.mutator)
        else:
            if all([isinstance(a, numpy.ndarray) for a in (X, y) if a is not None]):
                return DenseDesignMatrix(X=X, y=y, mutator=self.mutator)
            else:
                return SparseDesignMatrix(X=X, y=y, mutator=self.mutator)

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

    def _train_layer(self, trainer, layer, dataset):
        # Bug in PyLearn2 that has some unicode channels, can't sort.
        layer.monitor.channels = {str(k): v for k, v in layer.monitor.channels.items()}
        best_valid_error = float("inf")

        for i in itertools.count(1):
            start = time.time()
            trainer.train(dataset=dataset)

            layer.monitor.report_epoch()
            layer.monitor()
            
            objective = layer.monitor.channels.get('objective', None)
            if objective:
                avg_valid_error = objective.val_shared.get_value()
                best_valid_error = min(best_valid_error, avg_valid_error)
            else:
                # 'objective' channel is only defined with validation set.
                avg_valid_error = None

            best_valid = bool(best_valid_error == avg_valid_error)
            log.debug("{:>5}      {}{}{}        {:>5.1f}s".format(
                      i,
                      ansi.GREEN if best_valid else "",
                      "{:>10.6f}".format(float(avg_valid_error)) if (avg_valid_error is not None) else "     N/A  ",
                      ansi.ENDC if best_valid else "",
                      time.time() - start
                      ))

            if not trainer.continue_learning(layer):
                log.debug("")
                log.info("Early termination condition fired at %i iterations.", i)
                break
            if self.n_iter is not None and i >= self.n_iter:
                log.debug("")
                log.info("Terminating after specified %i total iterations.", i)
                break
