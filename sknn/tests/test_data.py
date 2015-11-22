import random
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
                    batch_size=1,
                    callback={'on_batch_start': self._mutate_fn})

    def _mutate_fn(self, Xb, **_):
        self.called += 1
        Xb[Xb == 0.0] = self.value

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

    def test_SetLayerParamsList(self):
        nn = MLPR(layers=[L("Linear")])
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn._initialize(a_in, a_out)
        
        weights = numpy.random.uniform(-1.0, +1.0, (16,4))
        biases = numpy.random.uniform(-1.0, +1.0, (4,))
        nn.set_parameters([(weights, biases)])
        
        p = nn.get_parameters()
        assert_true((p[0].weights.astype('float32') == weights.astype('float32')).all())
        assert_true((p[0].biases.astype('float32') == biases.astype('float32')).all())

    def test_LayerParamsSkipOneWithNone(self):
        nn = MLPR(layers=[L("Sigmoid", units=32), L("Linear", name='abcd')])
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn._initialize(a_in, a_out)
        
        weights = numpy.random.uniform(-1.0, +1.0, (32,4))
        biases = numpy.random.uniform(-1.0, +1.0, (4,))
        nn.set_parameters([None, (weights, biases)])
        
        p = nn.get_parameters()
        assert_true((p[1].weights.astype('float32') == weights.astype('float32')).all())
        assert_true((p[1].biases.astype('float32') == biases.astype('float32')).all())

    def test_SetLayerParamsDict(self):
        nn = MLPR(layers=[L("Sigmoid", units=32), L("Linear", name='abcd')])
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn._initialize(a_in, a_out)
        
        weights = numpy.random.uniform(-1.0, +1.0, (32,4))
        biases = numpy.random.uniform(-1.0, +1.0, (4,))
        nn.set_parameters({'abcd': (weights, biases)})
        
        p = nn.get_parameters()
        assert_true((p[1].weights.astype('float32') == weights.astype('float32')).all())
        assert_true((p[1].biases.astype('float32') == biases.astype('float32')).all())


class TestMaskedDataset(unittest.TestCase):

    def test_SingleOutputOne(self):
        nn = MLPR(layers=[L("Linear")], learning_rule='adam', n_iter=50)
        a_in, a_out, a_mask = numpy.random.uniform(-1.0, +1.0, (8,16)), numpy.zeros((8,1)), numpy.ones((8,))
        for i in range(8):
            if random.choice([True, False]):
                a_out[i] = 1.0
                a_mask[i] = 1.0
            else:
                a_out[i] = 0.0
                a_mask[i] = 0.0

        nn.fit(a_in, a_out, a_mask)
        v_out = nn.predict(a_in)
        
        # Make sure the examples weighted 1.0 have low error, 0.0 high error.
        print(abs(a_out - v_out).T[0] * (1.0 - a_mask))
        assert_true((abs(a_out - v_out).T[0] * a_mask < 5E-2).all())
        assert_true((abs(a_out - v_out).T[0] * (1.0 - a_mask) > 5E-1).any())

    def test_SingleOutputZero(self):
        nn = MLPR(layers=[L("Linear")], learning_rule='adam', n_iter=50)
        a_in, a_out, a_mask = numpy.random.uniform(-1.0, +1.0, (8,16)), numpy.zeros((8,1)), numpy.ones((8,))
        for i in range(8):
            if random.choice([True, False]):
                a_out[i] = 1.0
                a_mask[i] = 0.0
            else:
                a_out[i] = 0.0
                a_mask[i] = 1.0

        nn.fit(a_in, a_out, a_mask)
        v_out = nn.predict(a_in)
        
        # Make sure the examples weighted 1.0 have low error, 0.0 high error.
        print(abs(a_out - v_out).T[0] * (1.0 - a_mask))
        assert_true((abs(a_out - v_out).T[0] * a_mask < 5E-2).all())
        assert_true((abs(a_out - v_out).T[0] * (1.0 - a_mask) > 5E-1).any())

    def test_SingleOutputNegative(self):
        nn = MLPR(layers=[L("Linear")], learning_rule='adam', n_iter=50)
        a_in, a_out, a_mask = numpy.random.uniform(-1.0, +1.0, (8,16)), numpy.zeros((8,1)), numpy.ones((8,))
        for i in range(8):
            if random.choice([True, False]):
                a_out[i] = -1.0
                a_mask[i] = 1.0
            else:
                a_out[i] = 0.0
                a_mask[i] = 0.0

        nn.fit(a_in, a_out, a_mask)
        v_out = nn.predict(a_in)
        
        # Make sure the examples weighted 1.0 have low error, 0.0 high error.
        print(abs(a_out - v_out).T[0] * (1.0 - a_mask))
        assert_true((abs(a_out - v_out).T[0] * (1.0 - a_mask) > 5E-1).any())
        assert_true((abs(a_out - v_out).T[0] * a_mask < 5E-2).all())