import unittest
from nose.tools import (assert_false, assert_raises, assert_true, assert_equal)

import io
import pickle
import numpy
from sklearn.base import clone

from sknn.mlp import MultiLayerPerceptronRegressor as MLPR
from . import test_linear


class TestDeepNetwork(test_linear.TestLinearNetwork):

    def setUp(self):
        self.nn = MLPR(
            layers=[
                ("Rectifier", 16),
                ("Sigmoid", 12),
                ("Maxout", 8, 2),
                ("Tanh", 4),
                ("Linear",)],
            n_iter=1)

    def test_UnknownOuputActivation(self):
        nn = MLPR(layers=[("Unknown", 8)])
        a_in = numpy.zeros((8,16))
        assert_raises(NotImplementedError, nn.fit, a_in, a_in)

    def test_UnknownHiddenActivation(self):
        nn = MLPR(layers=[("Unknown", 8), ("Linear",)])
        a_in = numpy.zeros((8,16))
        assert_raises(NotImplementedError, nn.fit, a_in, a_in)

    # This class also runs all the tests from the linear network too.


class TestDeepDeterminism(unittest.TestCase):

    def setUp(self):
        self.a_in = numpy.random.uniform(0.0, 1.0, (8,16))
        self.a_out = numpy.zeros((8,1))

    def run_EqualityTest(self, copier, asserter):
        for activation in ["Rectifier", "Sigmoid", "Maxout", "Tanh"]:
            nn1 = MLPR(layers=[(activation, 16, 2), ("Linear", 8)], random_state=1234)
            nn1._initialize(self.a_in, self.a_out)

            nn2 = copier(nn1, activation)
            asserter(numpy.all(nn1.predict(self.a_in) == nn2.predict(self.a_in)))

    def test_DifferentSeedPredictNotEquals(self):
        def ctor(_, activation):
            nn = MLPR(layers=[(activation, 16, 2), ("Linear", 8)], random_state=2345)
            nn._initialize(self.a_in, self.a_out)
            return nn
        self.run_EqualityTest(ctor, assert_false)

    def test_SameSeedPredictEquals(self):
        def ctor(_, activation):
            nn = MLPR(layers=[(activation, 16, 2), ("Linear", 8)], random_state=1234)
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
