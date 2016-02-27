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
from sknn.mlp import Layer as L, Convolution as C

from . import test_linear


class TestDeepNetwork(test_linear.TestLinearNetwork):

    def setUp(self):
        self.nn = MLPR(
            layers=[
                L("Rectifier", units=16),
                L("Sigmoid", dropout=0.2, units=12),
                L("ExpLin", weight_decay=0.001, units=8),
                L("Tanh", normalize='batch', units=4),
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

    def run_EqualityTest(self, maker, copier, asserter):
        for activation in ["Rectifier", "Sigmoid", "Tanh", "ExpLin"]:
            nn1 = maker(activation)
            nn2 = copier(nn1, activation)
            print('activation', activation)
            a_out1 = nn1.predict(self.a_in)
            a_out2 = nn2.predict(self.a_in)
            print(a_out1, a_out2)
            asserter(numpy.all(nn1.predict(self.a_in) - nn2.predict(self.a_in) < 1E-6))

    def make(self, activation, seed=1234, train=False, **keywords):
        nn = MLPR(layers=[L(activation, units=16, **keywords),
                          L("Linear", units=1)], random_state=seed, n_iter=1)
        if train:
            nn.fit(self.a_in, self.a_out)
        else:
            nn._initialize(self.a_in, self.a_out)
        return nn 

    def test_DifferentSeedPredictNotEquals(self):
        for t in [True, False]:
            self.run_EqualityTest(lambda a: self.make(a, train=t),
                                lambda _, a: self.make(a, seed=2345, train=t), assert_false)

    def test_SameSeedPredictEquals(self):
        for t in [True, False]:
            self.run_EqualityTest(lambda a: self.make(a, train=t),
                                  lambda _, a: self.make(a, train=t), assert_true)

    def cloner(self, nn, _):
        cc = clone(nn)
        cc._initialize(self.a_in, self.a_out)
        return cc

    def test_ClonePredictEquals(self):
        self.run_EqualityTest(lambda a: self.make(a, train=False), self.cloner, assert_true)

    def serialize(self, nn, _):
        buf = io.BytesIO()
        pickle.dump(nn, buf)
        buf.seek(0)
        return pickle.load(buf)

    def test_SerializedPredictEquals(self):
        for t in [True, False]:
            self.run_EqualityTest(lambda a: self.make(a, train=t), self.serialize, assert_true)

    def test_BatchNormSerializePredictEquals(self):
        self.run_EqualityTest(lambda a: self.make(a, train=True, normalize='batch'), self.serialize, assert_true)

    def test_DropoutSerializePredictEquals(self):
        self.run_EqualityTest(lambda a: self.make(a, train=True, dropout=0.5), self.serialize, assert_true)


class TestConvolutionDeterminism(TestDeepDeterminism):

    def setUp(self):
        self.a_in = numpy.random.uniform(0.0, 1.0, size=(16,4,4,1))
        self.a_out = numpy.zeros((16,1))

    def make(self, activation, seed=1234, train=False, **keywords):
        nn = MLPR(layers=[C(activation, channels=16, kernel_shape=(3,3), **keywords), L("Linear")], random_state=seed, n_iter=1)
        if train:
            nn.fit(self.a_in, self.a_out)
        else:
            nn._initialize(self.a_in, self.a_out)
        return nn 


"""
TODO: [alexjc] Reintroduce with LRELU support.

class TestActivations(unittest.TestCase):

    def setUp(self):
        self.buf = io.StringIO()
        self.hnd = logging.StreamHandler(self.buf)
        logging.getLogger('sknn').addHandler(self.hnd)
        logging.getLogger().setLevel(logging.WARNING)

    def tearDown(self):
        assert_equal('', self.buf.getvalue())
        sknn.mlp.log.removeHandler(self.hnd)

    def test_UnusedParameterWarning(self):
        nn = MLPR(layers=[L("Linear", alpha=1E-3)], n_iter=1)
        a_in = numpy.zeros((8,16))
        nn._initialize(a_in, a_in)

        assert_in('Parameter `alpha` is unused', self.buf.getvalue())
        self.buf = io.StringIO() # clear
"""
