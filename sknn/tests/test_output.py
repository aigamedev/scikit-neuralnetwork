import unittest
from nose.tools import (assert_is_not_none, assert_raises, assert_equal)

import numpy

import sknn
from sknn.mlp import Regressor as MLPR
from sknn.mlp import Layer as L

from . import test_linear


class TestSoftmaxOutput(test_linear.TestLinearNetwork):

    def setUp(self):
        self.nn = MLPR(layers=[L("Softmax")], n_iter=1)


class TestLossTypes(unittest.TestCase):

    def test_UnknownLossType(self):
        assert_raises(AssertionError, MLPR, layers=[], loss_type='unknown')

    def test_MeanSquaredErrorLinear(self):
        nn = MLPR(layers=[L("Linear")], loss_type='mse', n_iter=1)
        self._run(nn)
    
    @unittest.skipIf(sknn.backend.name != 'lasagne', 'only lasagne')
    def test_CategoricalCrossEntropyLinear(self):
        nn = MLPR(layers=[L("Softmax")], loss_type='mcc', n_iter=1)
        self._run(nn)

    def _run(self, nn):
        a_in, a_out = numpy.ones((8,16)), numpy.ones((8,4))
        nn.fit(a_in, a_out)
        a_test = nn.predict(a_in)
