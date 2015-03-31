import unittest

import io
import pickle
import numpy as np

from nose.tools import (assert_is_not_none)
from sklearn.utils.testing import (assert_raises, assert_equal)


from sknn import NeuralNetwork


class TestDeepNetwork(unittest.TestCase):

    def setUp(self):
        self.nn = NeuralNetwork(
            layers=[
                ("RectifiedLinear", 16),
                ("Sigmoid", 12),
                ("Maxout", 8, 2),
                ("Tanh", 4),
                ("Linear",),
            ])

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
        self.nn.initialize(a_in, a_in)
        a_out = self.nn.predict(a_in)
        assert_equal(type(a_out), type(a_in))

    def test_FitAutoInitialize(self):
        a_in, a_out = np.zeros((8,16)), np.zeros((8,4))
        self.nn.fit(a_in, a_out)

    def test_FitWrongSize(self):
        a_in, a_out = np.zeros((7,16)), np.zeros((9,4))
        assert_raises(AssertionError, self.nn.fit, a_in, a_out)
