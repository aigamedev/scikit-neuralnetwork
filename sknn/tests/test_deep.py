import unittest
from nose.tools import (assert_false, assert_raises, assert_true,
                        assert_equal, assert_in)

import io
import pickle
import numpy
import logging

from sklearn.base import clone

import sknn
from sknn.mlp import Regressor as MLPR
from sknn.mlp import Layer as L

from . import test_linear


class TestDeepNetwork(test_linear.TestLinearNetwork):

    def setUp(self):
        self.nn = MLPR(
            layers=[
                L("Rectifier", units=16),
                L("Sigmoid", units=12),
                L("Maxout", units=16, pieces=2),
                L("Tanh", units=4),
                L("Linear")],
            n_iter=1)

    def test_UnknownLayer(self):
        assert_raises(NotImplementedError, L, "Unknown")

    def test_UnknownActivation(self):
        assert_raises(NotImplementedError, L, "Wrong", units=16)

    # This class also runs all the tests from the linear network too.


class TestDeepDeterminism(unittest.TestCase):

    def setUp(self):
        self.a_in = numpy.random.uniform(0.0, 1.0, (8,16))
        self.a_out = numpy.zeros((8,1))

    def run_EqualityTest(self, copier, asserter):
        for activation in ["Rectifier", "Sigmoid", "Maxout", "Tanh"]:
            nn1 = MLPR(layers=[L(activation, units=16, pieces=2), L("Linear", units=1)], random_state=1234)
            nn1._initialize(self.a_in, self.a_out)

            nn2 = copier(nn1, activation)
            asserter(numpy.all(nn1.predict(self.a_in) == nn2.predict(self.a_in)))

    def test_DifferentSeedPredictNotEquals(self):
        def ctor(_, activation):
            nn = MLPR(layers=[L(activation, units=16, pieces=2), L("Linear", units=1)], random_state=2345)
            nn._initialize(self.a_in, self.a_out)
            return nn
        self.run_EqualityTest(ctor, assert_false)

    def test_SameSeedPredictEquals(self):
        def ctor(_, activation):
            nn = MLPR(layers=[L(activation, units=16, pieces=2), L("Linear", units=1)], random_state=1234)
            nn._initialize(self.a_in, self.a_out)
            return nn
        self.run_EqualityTest(ctor, assert_true)

    def test_ClonePredictEquals(self):
        def cloner(nn, _):
            cc = clone(nn)
            cc._initialize(self.a_in, self.a_out)
            return cc
        self.run_EqualityTest(cloner, assert_true)

    def test_SerializedPredictEquals(self):
        def serialize(nn, _):
            buf = io.BytesIO()
            pickle.dump(nn, buf)
            buf.seek(0)
            return pickle.load(buf)
        self.run_EqualityTest(serialize, assert_true)


class TestActivations(unittest.TestCase):

    def setUp(self):
        self.buf = io.StringIO()
        self.hnd = logging.StreamHandler(self.buf)
        logging.getLogger('sknn').addHandler(self.hnd)
        logging.getLogger().setLevel(logging.WARNING)

    def tearDown(self):
        assert_equal('', self.buf.getvalue())
        sknn.mlp.log.removeHandler(self.hnd)

    def test_MissingParameterException(self):
        nn = MLPR(layers=[L("Maxout", units=32), L("Linear")])
        a_in = numpy.zeros((8,16))
        assert_raises(ValueError, nn._initialize, a_in, a_in)

    def test_UnusedParameterWarning(self):
        nn = MLPR(layers=[L("Linear", pieces=2)], n_iter=1)
        a_in = numpy.zeros((8,16))
        nn._initialize(a_in, a_in)

        assert_in('Parameter `pieces` is unused', self.buf.getvalue())
        self.buf = io.StringIO() # clear

