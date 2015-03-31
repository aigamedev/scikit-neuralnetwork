import unittest
import numpy as np

from sklearn.utils.testing import assert_raises, assert_equal

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
