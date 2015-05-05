import unittest
from nose.tools import (assert_is_not_none, assert_false, assert_raises,
                        assert_equal, assert_true)

import io
import pickle
import numpy

from sknn.mlp import Regressor as MLPR
from sknn.mlp import Layer as L


class TestLinearNetwork(unittest.TestCase):

    def setUp(self):
        self.nn = MLPR(layers=[L("Linear")], n_iter=1)

    def test_LifeCycle(self):
        del self.nn

    def test_PredictNoOutputUnitsAssertion(self):
        a_in = numpy.zeros((8,16))
        assert_raises(AssertionError, self.nn.predict, a_in)

    def test_AutoInitializeWithOutputUnits(self):
        self.nn.layers[-1].units = 4
        a_in = numpy.zeros((8,16))
        self.nn.predict(a_in)

    def test_FitAutoInitialize(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        self.nn.fit(a_in, a_out)
        assert_true(self.nn.is_initialized)

    def test_FitWrongSize(self):
        a_in, a_out = numpy.zeros((7,16)), numpy.zeros((9,4))
        assert_raises(AssertionError, self.nn.fit, a_in, a_out)


class TestInputOutputs(unittest.TestCase):

    def setUp(self):
        self.nn = MLPR(layers=[L("Linear")], n_iter=1)

    def test_FitOneDimensional(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,))
        self.nn.fit(a_in, a_out)


class TestSerialization(unittest.TestCase):

    def setUp(self):
        self.nn = MLPR(layers=[L("Linear")], n_iter=1)

    def test_SerializeFail(self):
        buf = io.BytesIO()
        assert_raises(AssertionError, pickle.dump, self.nn, buf)

    def test_SerializeCorrect(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        self.nn.fit(a_in, a_out)

        buf = io.BytesIO()
        pickle.dump(self.nn, buf)

        buf.seek(0)
        nn = pickle.load(buf)

        assert_is_not_none(nn.mlp)
        assert_equal(nn.layers, self.nn.layers)


class TestSerializedNetwork(TestLinearNetwork):

    def setUp(self):
        self.original = MLPR(layers=[L("Linear")])
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        self.original._initialize(a_in, a_out)

        buf = io.BytesIO()
        pickle.dump(self.original, buf)
        buf.seek(0)
        self.nn = pickle.load(buf)

    def test_TypeOfWeightsArray(self):
        for w, b in self.nn._mlp_to_array():
            assert_equal(type(w), numpy.ndarray)
            assert_equal(type(b), numpy.ndarray)

    def test_FitAutoInitialize(self):
        # Override base class test, you currently can't re-train a network that
        # was serialized and deserialized.
        pass

    def test_PredictNoOutputUnitsAssertion(self):
        # Override base class test, this is not initialized but it
        # should be able to predict without throwing assert.
        assert_true(self.nn.is_initialized)

    def test_PredictAlreadyInitialized(self):
        a_in = numpy.zeros((8,16))
        self.nn.predict(a_in)
