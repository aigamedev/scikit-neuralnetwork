import unittest
from nose.tools import (assert_true, assert_in, assert_raises, assert_equal)

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
        self.hnd.setLevel(logging.WARNING)
        logging.getLogger('sknn').addHandler(self.hnd)

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

    def test_adagrad(self):
         self._run(MLPR(layers=[L("Linear",)],
                        learning_rule='adagrad',
                        n_iter=1))

    def test_AdaDelta(self):
        self._run(MLPR(layers=[L("Linear")],
                       learning_rule='adadelta',
                       n_iter=1))

    def test_RmsProp(self):
        self._run(MLPR(layers=[L("Linear")],
                       learning_rule='rmsprop',
                       n_iter=1))

    def test_UnknownRule(self):
        nn = MLPR(layers=[L("Linear")], learning_rule='unknown')
        assert_raises(NotImplementedError, self._run, nn)


class TestRegularization(LoggingTestCase):

    def setUp(self):
        super(TestRegularization, self).setUp()
        self.output = io.StringIO()
        self.hnd2 = logging.StreamHandler(self.output)
        self.hnd2.setLevel(logging.DEBUG)
        logging.getLogger('sknn').addHandler(self.hnd2)

    def tearDown(self):
        super(TestRegularization, self).tearDown()
        sknn.mlp.log.removeHandler(self.hnd2)

    def test_DropoutExplicit(self):
        nn = MLPR(layers=[L("Tanh", units=8), L("Linear",)],
                  regularize='dropout',
                  n_iter=1)
        assert_equal(nn.regularize, 'dropout')
        self._run(nn)
        assert_in('Using `dropout` for regularization.', self.output.getvalue())

    def test_DropoutAsFloat(self):
        nn = MLPR(layers=[L("Tanh", units=8), L("Linear",)],
                  dropout_rate=0.25,
                  n_iter=1)
        assert_equal(nn.regularize, 'dropout')
        assert_equal(nn.dropout_rate, 0.25)
        self._run(nn)
        assert_in('Using `dropout` for regularization.', self.output.getvalue())

    def test_DropoutPerLayer(self):
        nn = MLPR(layers=[L("Rectifier", units=8, dropout=0.25), L("Linear")],
                  regularize='dropout',
                  n_iter=1)
        assert_equal(nn.regularize, 'dropout')
        self._run(nn)
        assert_in('Using `dropout` for regularization.', self.output.getvalue())

    def test_AutomaticDropout(self):
        nn = MLPR(layers=[L("Tanh", units=8, dropout=0.25), L("Linear")], n_iter=1)
        self._run(nn)
        assert_in('Using `dropout` for regularization.', self.output.getvalue())

    def test_RegularizeExplicitL1(self):
        nn = MLPR(layers=[L("Tanh", units=8), L("Linear",)],
                  regularize='L1',
                  n_iter=1)
        assert_equal(nn.regularize, 'L1')
        self._run(nn)
        assert_in('Using `L1` for regularization.', self.output.getvalue())

    def test_RegularizeExplicitL2(self):
        nn = MLPR(layers=[L("Sigmoid", units=8), L("Softmax",)],
                  regularize='L2',
                  n_iter=1)
        assert_equal(nn.regularize, 'L2')
        self._run(nn)
        assert_in('Using `L2` for regularization.', self.output.getvalue())

    def test_RegularizeCustomParam(self):
        nn = MLPR(layers=[L("Tanh", units=8), L("Linear",)],
                  weight_decay=0.01,
                  n_iter=1)
        assert_equal(nn.weight_decay, 0.01)
        self._run(nn)
        assert_in('Using `L2` for regularization.', self.output.getvalue())

    def test_RegularizePerLayer(self):
        nn = MLPR(layers=[L("Rectifier", units=8, weight_decay=0.01), L("Linear", weight_decay=0.001)],
                  n_iter=1)
        self._run(nn)
        assert_in('Using `L2` for regularization.', self.output.getvalue())

    def test_AutomaticRegularize(self):
        nn = MLPR(layers=[L("Tanh", units=8, weight_decay=0.0001), L("Linear")], n_iter=1)
        self._run(nn)
        assert_in('Using `L2` for regularization.', self.output.getvalue())
