import unittest
from nose.tools import (assert_true, assert_raises, assert_equal)

import io
import numpy
import logging

import sknn
from sknn.mlp import Regressor as MLPR
from sknn.mlp import Layer as L


class LoggingTestCase(unittest.TestCase):

    def setUp(self):
        self.buf = io.StringIO()
        self.hnd = logging.StreamHandler(self.buf)
        logging.getLogger('sknn').addHandler(self.hnd)
        logging.getLogger().setLevel(logging.WARNING)

    def tearDown(self):
        assert_equal('', self.buf.getvalue())
        sknn.mlp.log.removeHandler(self.hnd)

    def _run(self, nn):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn.fit(a_in, a_out)
        a_test = nn.predict(a_in)
        assert_equal(type(a_out), type(a_test))


class TestLearningRules(LoggingTestCase):

    def test_Default(self):
        self._run(MLPR(layers=[L("Linear")],
                       learning_rule='sgd',
                       n_iter=1))

    def test_Momentum(self):
        self._run(MLPR(layers=[L("Gaussian")],
                       learning_rule='momentum',
                       n_iter=1))

    def test_Nesterov(self):
        self._run(MLPR(layers=[L("Softmax")],
                       learning_rule='nesterov',
                       n_iter=1))

    # NOTE: This is currentry broken in PyLearn2.
    # def test_adagrad(self):
    #      self._run(MLPR(layers=[("Linear",)],
    #                     learning_rule='adagrad',
    #                     learning_rate=0.000001,
    #                     batch_size=100,
    #                     n_iter=1))

    def test_AdaDelta(self):
        self._run(MLPR(layers=[L("Linear")],
                       learning_rule='adadelta',
                       n_iter=1))

    def test_RmsProp(self):
        self._run(MLPR(layers=[L("Linear")],
                       learning_rule='rmsprop',
                       n_iter=1))

    def test_UnknownRule(self):
        assert_raises(NotImplementedError, MLPR,
                      layers=[], learning_rule='unknown')


class TestRegularization(LoggingTestCase):

    def test_DropoutAsBool(self):
        nn = MLPR(layers=[L("Tanh", units=8), L("Linear",)],
                  regularize='dropout',
                  n_iter=1)
        assert_equal(nn.regularize, 'dropout')
        assert_equal(nn.dropout_rate, 0.5)
        self._run(nn)
        assert_true(nn.cost is not None)

    def test_DropoutAsFloat(self):
        nn = MLPR(layers=[L("Tanh", units=8), L("Linear",)],
                  dropout_rate=0.25,
                  n_iter=1)
        assert_equal(nn.regularize, 'dropout')
        assert_equal(nn.dropout_rate, 0.25)
        self._run(nn)
        assert_true(nn.cost is not None)

    def test_DropoutPerLayer(self):
        nn = MLPR(layers=[L("Maxout", units=8, pieces=2, dropout=0.25), L("Linear")],
                  regularize='dropout',
                  n_iter=1)
        assert_equal(nn.regularize, 'dropout')
        assert_equal(nn.dropout_rate, 0.5)
        self._run(nn)
        assert_true(nn.cost is not None)

    def test_AutomaticDropout(self):
        nn = MLPR(layers=[L("Tanh", units=8, dropout=0.25), L("Linear")], n_iter=1)
        self._run(nn)
        assert_true(nn.cost is not None)