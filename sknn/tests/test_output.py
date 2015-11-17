import unittest
from nose.tools import (assert_is_not_none, assert_raises, assert_equal)

import numpy

import sknn
from sknn.mlp import Regressor as MLPR
from sknn.mlp import Layer as L

from . import test_linear


@unittest.skipIf(sknn.backend.name != 'pylearn2', 'only pylearn2')
class TestGaussianOutput(test_linear.TestLinearNetwork):

    def setUp(self):
        self.nn = MLPR(layers=[L("Gaussian")], n_iter=1)


class TestSoftmaxOutput(test_linear.TestLinearNetwork):

    def setUp(self):
        self.nn = MLPR(layers=[L("Softmax")], n_iter=1)


class TestLossTypes(unittest.TestCase):

    def test_UnknownLossType(self):
        assert_raises(AssertionError, MLPR, layers=[], loss_type='unknown')

    def test_MeanAverageErrorLinear(self):
        nn = MLPR(layers=[L("Linear")], loss_type='mae', n_iter=1)
        self._run(nn)

    def test_MeanSquaredErrorLinear(self):
        nn = MLPR(layers=[L("Linear")], loss_type='mse', n_iter=1)
        self._run(nn)

    @unittest.skipIf(sknn.backend.name != 'pylearn2', 'only pylearn2')
    def test_MeanAverageErrorGaussian(self):
        nn = MLPR(layers=[L("Gaussian")], loss_type='mae', n_iter=1)
        self._run(nn)

    @unittest.skipIf(sknn.backend.name != 'pylearn2', 'only pylearn2')
    def test_MeanSquaredErrorGaussian(self):
        nn = MLPR(layers=[L("Gaussian")], loss_type='mse', n_iter=1)
        self._run(nn)

    def _run(self, nn):
        a_in, a_out = numpy.ones((8,16)), numpy.ones((8,4))
        nn.fit(a_in, a_out)
        a_test = nn.predict(a_in)
