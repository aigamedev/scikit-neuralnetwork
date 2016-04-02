import unittest
from nose.tools import (assert_is_not_none, assert_true, assert_raises, assert_equal)

import io
import pickle
import numpy

from sknn.mlp import Regressor as MLPR
from sknn.mlp import Native as N, Layer as L

import lasagne.layers as ly
import lasagne.nonlinearities as nl


class TestNativeLasagneLayer(unittest.TestCase):

    def _run(self, nn):
        a_in, a_out = numpy.ones((8,16)), numpy.ones((8,4))
        nn.fit(a_in, a_out)

    def test_DenseLinear(self):
        nn = MLPR(layers=[N(ly.DenseLayer, num_units=4, nonlinearity=nl.linear)], n_iter=1)
        self._run(nn)

    def test_GaussianNoise(self):
        nn = MLPR(layers=[L("Rectifier", units=12), N(ly.GaussianNoiseLayer), L("Linear")], n_iter=1)
        self._run(nn)

    def test_LeakyRectifier(self):
        nn = MLPR(layers=[N(ly.DenseLayer, units=24, nonlinearity=nl.leaky_rectify),
                          L("Linear")], n_iter=1)
        self._run(nn)
