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
        activation = "Gaussian" if sknn.backend.name == "pylearn2" else "Linear"
        self.nn = MLP(
                    layers=[L(activation)], learning_rate=0.001,
                    n_iter=None, n_stable=1, f_stable=0.01,
                    valid_set=(a_in, a_out))

        self.nn._fit(a_in, a_out)

    def test_FitAutomaticValidation(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        self.nn = MLP(
                    layers=[L("Linear")], learning_rate=0.001,
                    n_iter=10, n_stable=1, f_stable=0.1,
                    valid_size=0.25)

        self.nn._fit(a_in, a_out)
        
    def test_TrainingInfinite(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        self.nn = MLP(layers=[L("Linear")], n_iter=None, n_stable=None)
        assert_raises(AssertionError, self.nn._fit, a_in, a_out)

    def test_TrainingUserDefined(self):
        self.counter = 0
        
        def terminate(**_):
            self.counter += 1
            return False

        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        self.nn = MLP(layers=[L("Linear")], n_iter=100, n_stable=None, callback={'on_epoch_finish': terminate})
        self.nn._fit(a_in, a_out)

        assert_equals(self.counter, 1)


class TestBatchSize(unittest.TestCase):

    def setUp(self):
        self.batch_count = 0
        self.batch_items = 0
        self.nn = MLP(
                    layers=[L("Rectifier")],
                    learning_rate=0.001, n_iter=1,
                    callback={'on_batch_start': self.on_batch_start})

    def on_batch_start(self, Xb, **args):
        self.batch_count += 1
        self.batch_items += Xb.shape[0]
        assert Xb.shape[0] <= self.nn.batch_size

    def test_BatchSizeLargerThanInput(self):
        self.nn.batch_size = 32
        a_in, a_out = numpy.zeros((8,16)), numpy.ones((8,4))
        self.nn._fit(a_in, a_out)
        assert_equals(1, self.batch_count)
        assert_equals(8, self.batch_items)

    def test_BatchSizeSmallerThanInput(self):
        self.nn.batch_size = 4
        a_in, a_out = numpy.ones((8,16)), numpy.zeros((8,4))
        self.nn._fit(a_in, a_out)
        assert_equals(2, self.batch_count)
        assert_equals(8, self.batch_items)

    def test_BatchSizeNonMultiple(self):
        self.nn.batch_size = 4
        a_in, a_out = numpy.zeros((9,16)), numpy.ones((9,4))
        self.nn._fit(a_in, a_out)
        assert_equals(3, self.batch_count)
        assert_equals(9, self.batch_items)


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
        assert_in("Epoch       Training Error       Validation Error       Time", self.buf.getvalue())
        assert_in("    1       ", self.buf.getvalue())
        assert_in("    N/A     ", self.buf.getvalue())

    def test_VerboseClassifier(self):
        nn = MLPC(layers=[L("Softmax")], verbose=1, n_iter=1)
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,1), dtype=numpy.int32)
        nn.fit(a_in, a_out)
        assert_in("Epoch       Training Error       Validation Error       Time", self.buf.getvalue())
        assert_in("    1       ", self.buf.getvalue())
        assert_in("    N/A     ", self.buf.getvalue())

    def test_CaughtRuntimeError(self):
        nn = MLPC(layers=[L("Linear")], learning_rate=float("nan"), n_iter=1)
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,1), dtype=numpy.int32)
        assert_raises(RuntimeError, nn.fit, a_in, a_out)
        assert_in("A runtime exception was caught during training.", self.buf.getvalue())
