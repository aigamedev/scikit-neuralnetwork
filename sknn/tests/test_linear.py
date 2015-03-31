import unittest

import io
import pickle
import numpy as np

from nose.tools import (assert_is_not_none)
from sklearn.utils.testing import (assert_raises, assert_equal)


from sknn import NeuralNetwork


class TestLinearNetwork(unittest.TestCase):

    def setUp(self):
        self.nn = NeuralNetwork(layers=[("Linear",)])

    def test_LifeCycle(self):
        del self.nn

    def test_InitializeManually(self):
        a_in, a_out = np.zeros((8,16)), np.zeros((8,4))
        self.nn.initialize(a_in, a_out)

    def test_PredictUninitialized(self):
        a_in = np.zeros((8,16))
        assert_raises(AssertionError, self.nn.predict, a_in)

    def test_PredictAutoInitialize(self):
        a_in = np.zeros((8,16))
        a_out = self.nn.predict(a_in, 4)
        assert_equal(type(a_out), type(a_in))

    def test_FitAutoInitialize(self):
        a_in, a_out = np.zeros((8,16)), np.zeros((8,4))
        self.nn.fit(a_in, a_out)

    def test_FitWrongSize(self):
        a_in, a_out = np.zeros((7,16)), np.zeros((9,4))
        assert_raises(AssertionError, self.nn.fit, a_in, a_out)


class TestSerialization(unittest.TestCase):

    def setUp(self):
        self.nn = NeuralNetwork(layers=[("Linear",)])

    def test_SerializeFail(self):
        buf = io.BytesIO()
        assert_raises(AssertionError, pickle.dump, self.nn, buf)

    def test_SerializeCorrect(self):
        a_in, a_out = np.zeros((8,16)), np.zeros((8,4))
        self.nn.initialize(a_in, a_out)

        buf = io.BytesIO()
        pickle.dump(self.nn, buf)

        buf.seek(0)
        nn = pickle.load(buf)

        assert_is_not_none(nn.mlp)
        assert_equal(nn.layers, self.nn.layers)


class TestSerializedNetwork(TestLinearNetwork):

    def setUp(self):
        self.original = NeuralNetwork(layers=[("Linear",)])
        a_in, a_out = np.zeros((8,16)), np.zeros((8,4))
        self.original.initialize(a_in, a_out)

        buf = io.BytesIO()
        pickle.dump(self.original, buf)
        buf.seek(0)
        self.nn = pickle.load(buf)
