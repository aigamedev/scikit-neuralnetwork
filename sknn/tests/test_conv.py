import unittest
from nose.tools import (assert_is_not_none, assert_raises, assert_equal)

import numpy

from sknn.mlp import MultiLayerPerceptronRegressor as MLPR
from sknn.mlp import Layer as L


class TestConvolution(unittest.TestCase):

    def _run(self, nn):
        a_in, a_out = numpy.zeros((8,32,16,1)), numpy.zeros((8,4))
        nn.fit(a_in, a_out)
        a_test = nn.predict(a_in)
        assert_equal(type(a_out), type(a_in))

    def test_SquareKernel(self):
        self._run(MLPR(
            layers=[
                L("Convolution", channels=4, shape=(2,2)),
                ("Linear",)],
            n_iter=1))

    def test_VerticalKernel(self):
        self._run(MLPR(
            layers=[
                L("Convolution", channels=4, shape=(16,1)),
                ("Linear",)],
            n_iter=1))


    def test_VerticalVerbose(self):
        self._run(MLPR(
            layers=[
                L("Convolution", channels=4, shape=(16,1)),
                ("Linear",)],
            n_iter=1, verbose=1, valid_size=0.1))

    def test_HorizontalKernel(self):
        self._run(MLPR(
            layers=[
                L("Convolution", channels=4, shape=(1,16)),
                ("Linear",)],
            n_iter=1))

    def test_ValidationSet(self):
        self._run(MLPR(
            layers=[
                L("Convolution", channels=4, shape=(2,2)),
                ("Linear",)],
            n_iter=1,
            valid_size=0.5))

    def test_MultipleLayers(self):
        self._run(MLPR(
            layers=[
                L("Convolution", channels=6, shape=(3,3)),
                L("Convolution", channels=4, shape=(5,5)),
                L("Convolution", channels=8, shape=(3,3)),
                ("Linear",)],
            n_iter=1))


class TestConvolutionRGB(TestConvolution):

    def _run(self, nn):
        a_in, a_out = numpy.zeros((8,32,16,3)), numpy.zeros((8,4))
        nn.fit(a_in, a_out)
        a_test = nn.predict(a_in)
        assert_equal(type(a_out), type(a_in))
