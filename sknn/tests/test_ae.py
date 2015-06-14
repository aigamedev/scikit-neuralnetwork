import unittest
from nose.tools import (assert_raises, assert_equals)

import numpy

import sknn
from sknn.ae import AutoEncoder as AE, Layer as L
from sknn import mlp


@unittest.skipIf(sknn.backend.name != 'pylearn2', 'only pylearn2')
class TestAutoEncoder(unittest.TestCase):
    
    def test_LifeCycle(self):
        ae = AE(layers=[L("Sigmoid", units=8)])
        del ae

    def test_FitData(self):
        X = numpy.zeros((8,4))
        ae = AE(layers=[L("Sigmoid", units=8)], n_iter=1)
        ae.fit(X)

    def test_FitVerbose(self):
        X = numpy.zeros((8,4))
        ae = AE(layers=[L("Sigmoid", units=8)], n_iter=1, verbose=1)
        ae.fit(X)

    def test_TransferSuccess(self):
        X = numpy.zeros((8,4))
        ae = AE(layers=[L("Tanh", units=4)], n_iter=1)
        ae.fit(X)

        nn = mlp.MultiLayerPerceptron(
                layers=[mlp.Layer("Tanh", units=4)])
        ae.transfer(nn)

    def test_TransferFailure(self):
        X = numpy.zeros((8,4))
        ae = AE(layers=[L("Tanh", units=8)], n_iter=1)
        ae.fit(X)

        nn = mlp.MultiLayerPerceptron(
                layers=[mlp.Layer("Tanh", units=4)])
        assert_raises(AssertionError, ae.transfer, nn)


@unittest.skipIf(sknn.backend.name != 'pylearn2', 'only pylearn2')
class TestParameters(unittest.TestCase):
    
    def test_CostFunctions(self):
        X = numpy.zeros((8,12))
        for t in ['msre', 'mbce']:
            ae = AE(layers=[L("Sigmoid", units=4, cost=t)], n_iter=1)
            y = ae.fit_transform(X)

            assert_equals(type(y), numpy.ndarray)
            assert_equals(y.shape, (8, 4))

    def test_LayerTypes(self):
        X = numpy.zeros((8,12))
        for l in ['autoencoder', 'denoising']:
            ae = AE(layers=[L("Sigmoid", type=l, units=4)])
            y = ae.fit_transform(X)

            assert_equals(type(y), numpy.ndarray)
            assert_equals(y.shape, (8, 4))

    def test_UnknownCostFunction(self):
        assert_raises(NotImplementedError, L, "Sigmoid", cost="unknown")

    def test_UnknownType(self):
        assert_raises(NotImplementedError, L, "Sigmoid", type="unknown")

    def test_UnknownActivation(self):
        assert_raises(NotImplementedError, L, "Unknown")
