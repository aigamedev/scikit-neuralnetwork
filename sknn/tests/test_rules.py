import unittest
from nose.tools import (assert_is_not_none, assert_false, assert_raises, assert_equal)

import numpy

from sknn.mlp import MultiLayerPerceptronRegressor as MLPR


class TestLearningRules(unittest.TestCase):

    def test_default(self):
        self._run(MLPR(layers=[("Linear",)],
                       learning_rule='sgd',
                       n_iter=1))

    def test_momentum(self):
        self._run(MLPR(layers=[("Linear",)],
                       learning_rule='momentum',
                       n_iter=1))

    def test_nesterov(self):
        self._run(MLPR(layers=[("Linear",)],
                       learning_rule='nesterov',
                       n_iter=1))

    # NOTE: This is currentry broken in PyLearn2.
    # def test_adagrad(self):
    #      self._run(MLPR(layers=[("Linear",)],
    #                     learning_rule='adagrad',
    #                     learning_rate=0.000001,
    #                     batch_size=100,
    #                     n_iter=1))

    def test_adadelta(self):
        self._run(MLPR(layers=[("Linear",)],
                       learning_rule='adadelta',
                       n_iter=1))

    def test_rmsprop(self):
        self._run(MLPR(layers=[("Linear",)],
                       learning_rule='rmsprop',
                       n_iter=1))

    def test_dropout(self):
        self._run(MLPR(layers=[("Sigmoid", 8), ("Linear",)],
                       dropout=True,
                       n_iter=1))

    def test_unknown(self):
        assert_raises(NotImplementedError, MLPR,
                      layers=[], learning_rule='unknown')

    def _run(self, nn):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn.fit(a_in, a_out)
        a_test = nn.predict(a_in)
        assert_equal(type(a_out), type(a_test))
