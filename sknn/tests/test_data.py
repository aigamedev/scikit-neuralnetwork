import unittest
from nose.tools import (assert_in, assert_raises, assert_equals, assert_true)

import logging

import numpy
from sknn.mlp import Regressor as MLPR
from sknn.mlp import Layer as L, Convolution as C


class TestDataAugmentation(unittest.TestCase):

    def setUp(self):
        self.called = 0
        self.value = 1.0

        self.nn = MLPR(
                    layers=[L("Linear")],
                    n_iter=1,
                    batch_size=2,
                    mutator=self._mutate_fn)

    def _mutate_fn(self, sample):
        self.called += 1
        sample[sample == 0.0] = self.value

    def test_TestCalledOK(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        self.nn._fit(a_in, a_out)
        assert_equals(a_in.shape[0], self.called)

    def test_DataIsUsed(self):
        self.value = float("nan")
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        assert_raises(RuntimeError, self.nn._fit, a_in, a_out)


class TestNetworkParameters(unittest.TestCase):
    
    def test_GetLayerParams(self):
        nn = MLPR(layers=[L("Linear")], n_iter=1)
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn._initialize(a_in, a_out)
        
        p = nn.get_parameters()
        assert_equals(type(p), list)
        assert_true(isinstance(p[0], tuple))
        
        assert_equals(p[0].layer, 'output')
        assert_equals(p[0].weights.shape, (16, 4))
        assert_equals(p[0].biases.shape, (4,))
