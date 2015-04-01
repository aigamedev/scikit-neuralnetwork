import unittest

import numpy as np

from nose.tools import (assert_is_not_none, assert_raises, assert_equal)

from sknn.nn import NeuralNetwork


class TestConvolution(unittest.TestCase):

    def _run(self, nn):
        a_in, a_out = np.zeros((8,16,16)), np.zeros((8,4))
        nn.fit(a_in, a_out)
        a_test = nn.predict(a_in)
        assert_equal(type(a_out), type(a_in))

    def test_SquareKernel(self):
        self._run(NeuralNetwork(
            layers=[
                ("Convolution", 4, (2,2)),
                ("Linear",),
            ]))

    def test_VerticalKernel(self):
        self._run(NeuralNetwork(
            layers=[
                ("Convolution", 4, (16,1)),
                ("Linear",),
            ]))

    def test_HorizontalKernel(self):
        self._run(NeuralNetwork(
            layers=[
                ("Convolution", 4, (1,16)),
                ("Linear",),
            ]))
