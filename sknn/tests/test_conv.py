import unittest
from nose.tools import (assert_is_not_none, assert_raises, assert_equal)

import numpy

from sknn.mlp import MultiLayerPerceptronRegressor as MLPR


class TestConvolution(unittest.TestCase):

    def _run(self, nn):
        a_in, a_out = numpy.zeros((8,16,16)), numpy.zeros((8,4))
        nn.fit(a_in, a_out)
        a_test = nn.predict(a_in)
        assert_equal(type(a_out), type(a_in))

    def test_SquareKernel(self):
        self._run(MLPR(
            layers=[
                ("Convolution", 4, (2,2)),
                ("Linear",)],
            n_iter=1))

    def test_VerticalKernel(self):
        self._run(MLPR(
            layers=[
                ("Convolution", 4, (16,1)),
                ("Linear",)],
            n_iter=1))

    def test_HorizontalKernel(self):
        self._run(MLPR(
            layers=[
                ("Convolution", 4, (1,16)),
                ("Linear",)],
            n_iter=1))
