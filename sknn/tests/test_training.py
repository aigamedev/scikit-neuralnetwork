import unittest
from nose.tools import (assert_in, assert_raises, assert_equals)

import io
import logging

import numpy
from sknn.mlp import MultiLayerPerceptron as MLP
from sknn.mlp import Regressor as MLPR
from sknn.mlp import Classifier as MLPC
from sknn.mlp import Layer as L

import sknn.mlp


class TestTrainingProcedure(unittest.TestCase):

    def test_FitTerminateStable(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        self.nn = MLP(
                    layers=[L("Gaussian")], learning_rate=0.001,
                    n_iter=None, n_stable=1, f_stable=0.1,
                    valid_set=(a_in, a_out))

        self.nn._fit(a_in, a_out)

    def test_FitAutomaticValidation(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        self.nn = MLP(
                    layers=[L("Gaussian")], learning_rate=0.001,
                    n_iter=10, n_stable=1, f_stable=0.1,
                    valid_size=0.25)

        self.nn._fit(a_in, a_out)


class TestCustomLogging(unittest.TestCase):

    def setUp(self):
        self.log = logging.getLogger('sknn')
        self.log.handlers = []
        self.backup, self.log.parent.handlers = self.log.parent.handlers, []

    def tearDown(self):
        self.log.parent.handlers = self.backup

    def test_DefaultLogVerbose(self):
        nn = MLPR(layers=[L("Linear")], verbose=True)
        assert_equals(1, len(self.log.handlers))
        assert_equals(logging.DEBUG, self.log.handlers[0].level)

    def test_DefaultLogQuiet(self):
        nn = MLPR(layers=[L("Linear")], verbose=False)
        assert_equals(1, len(self.log.handlers))
        assert_equals(logging.WARNING, self.log.handlers[0].level)

    def test_VerboseNoneNoLog(self):
        nn = MLPR(layers=[L("Linear")], verbose=None)
        assert_equals(0, len(self.log.handlers))


class TestTrainingOutput(unittest.TestCase):

    def setUp(self):
        self.buf = io.StringIO()
        self.hnd = logging.StreamHandler(self.buf)
        logging.getLogger('sknn').addHandler(self.hnd)

    def tearDown(self):
        sknn.mlp.log.removeHandler(self.hnd)

    def test_VerboseRegressor(self):
        nn = MLPR(layers=[L("Linear")], verbose=1, n_iter=1)
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn.fit(a_in, a_out)
        assert_in("Epoch    Validation Error      Time", self.buf.getvalue())
        assert_in("    1           N/A          ", self.buf.getvalue())

    def test_VerboseClassifier(self):
        nn = MLPC(layers=[L("Linear")], verbose=1, n_iter=1)
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,1), dtype=numpy.int32)
        nn.fit(a_in, a_out)
        assert_in("Epoch    Validation Error      Time", self.buf.getvalue())
        assert_in("    1           N/A          ", self.buf.getvalue())

    def test_CaughtRuntimeError(self):
        nn = MLPC(layers=[L("Linear")], learning_rate=float("nan"), n_iter=1)
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,1), dtype=numpy.int32)
        assert_raises(RuntimeError, nn.fit, a_in, a_out)
        assert_in("A runtime exception was caught during training.", self.buf.getvalue())
