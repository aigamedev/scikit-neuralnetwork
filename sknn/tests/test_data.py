import random
import unittest
from nose.tools import (assert_greater, assert_less, assert_raises,\
                        assert_equals, assert_in, assert_true)

import io
import logging

import numpy
from sknn.mlp import Regressor as MLPR, Classifier as MLPC
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
    
    def setUp(self):
        self.buf = io.StringIO()
        self.hnd = logging.StreamHandler(self.buf)
        logging.getLogger('sknn').addHandler(self.hnd)

    def tearDown(self):
        logging.getLogger('sknn').removeHandler(self.hnd)

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

    def test_SetParametersBeforeInit(self):
        nn = MLPR(layers=[L("Linear")])
        weights = numpy.random.uniform(-1.0, +1.0, (16,4))
        biases = numpy.random.uniform(-1.0, +1.0, (4,))
        nn.set_parameters([(weights, biases)])

        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn._initialize(a_in, a_out)
        assert_in('Reloading parameters for 1 layer weights and biases.', self.buf.getvalue())

    def test_SetParametersConstructor(self):
        weights = numpy.random.uniform(-1.0, +1.0, (16,4))
        biases = numpy.random.uniform(-1.0, +1.0, (4,))
        nn = MLPR(layers=[L("Linear")], parameters=[(weights, biases)])

        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn._initialize(a_in, a_out)
        assert_in('Reloading parameters for 1 layer weights and biases.', self.buf.getvalue())

    def test_GetParamsThenConstructor(self):
        nn1 = MLPR(layers=[L("Linear")], n_iter=1)
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn1._initialize(a_in, a_out)
        
        p1 = nn1.get_parameters()
        print(len(p1))
        nn2 = MLPR(layers=[L("Linear")], n_iter=1, parameters=p1)
        nn2._initialize(a_in, a_out)
        p2 = nn2.get_parameters()
        print(len(p2))
        
        assert_true((p1[0].weights.astype('float32') == p2[0].weights.astype('float32')).all())
        assert_true((p1[0].biases.astype('float32') == p2[0].biases.astype('float32')).all())

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


class TestMaskedDataRegression(unittest.TestCase):

    def check(self, a_in, a_out, a_mask):
        nn = MLPR(layers=[L("Linear")], learning_rule='adam', learning_rate=0.05, n_iter=250, n_stable=25)
        nn.fit(a_in, a_out, a_mask)
        v_out = nn.predict(a_in)

        # Make sure the examples weighted 1.0 have low error, 0.0 high error.
        masked = abs(a_out - v_out).T * a_mask
        print('masked', masked)
        assert_true((masked < 5.0E-1).all())
        inversed = abs(a_out - v_out).T * (1.0 - a_mask)
        print('inversed', inversed)
        assert_greater(inversed.mean(), masked.mean())

    def test_SingleOutputOne(self):
        a_out = numpy.random.randint(2, size=(8,1)).astype(numpy.float32)
        a_in = numpy.random.uniform(-1.0, +1.0, (8,16))
        a_mask = (0.0 + a_out).flatten()
        
        self.check(a_in, a_out, a_mask)

    def test_SingleOutputZero(self):
        a_in = numpy.random.uniform(-1.0, +1.0, (8,16))
        a_out = numpy.random.randint(2, size=(8,1)).astype(numpy.float32)
        a_out[0], a_out[-1] = 0, 1
        a_mask = (1.0 - a_out).flatten()

        self.check(a_in, a_out, a_mask)

    def test_SingleOutputNegative(self):
        a_in = numpy.random.uniform(-1.0, +1.0, (8,16))
        a_out = numpy.random.randint(2, size=(8,1)).astype(numpy.float32)
        a_out[0], a_out[-1] = 0, 1
        a_mask = (0.0 + a_out).flatten()
        a_out = -1.0 + 2.0 * a_out
        
        self.check(a_in, a_out, a_mask)
        
    def test_MultipleOutputRandom(self):
        a_in = numpy.random.uniform(-1.0, +1.0, (8,16))
        a_out = numpy.random.randint(2, size=(8,4)).astype(numpy.float32)
        a_out[0], a_out[-1] = 0, 1
        a_mask = (a_out.mean(axis=1) > 0.5).astype(numpy.float32)

        self.check(a_in, a_out, a_mask)


class TestMaskedDataClassification(unittest.TestCase):

    def check(self, a_in, a_out, a_mask, act='Softmax'):
        nn = MLPC(layers=[L(act)], learning_rule='adam', learning_rate=0.05, n_iter=250, n_stable=25)
        nn.fit(a_in, a_out, a_mask)
        return nn.predict_proba(a_in)

    def test_TwoLabelsOne(self):
        # Only one sample has the value 1 with weight 1.0, but all 0s are weighted 0.0.
        a_in = numpy.random.uniform(-1.0, +1.0, (16,4))
        a_out = numpy.zeros((16,1), dtype=numpy.int32)
        a_out[0] = 1
        a_mask = (0.0 + a_out).flatten().astype(numpy.float32)
        
        a_test = (self.check(a_in, a_out, a_mask) * a_mask.reshape((-1,1))).mean(axis=0)
        assert_greater(a_test[1], a_test[0] * 1.25)

    def test_TwoLabelsZero(self):
        # Only one sample has the value 0 with weight 1.0, but all 1s are weighted 0.0. 
        a_in = numpy.random.uniform(-1.0, +1.0, (16,4))
        a_out = numpy.ones((16,1), dtype=numpy.int32)
        a_out[-1] = 0
        a_mask = (1.0 - a_out).flatten()
        
        a_test = (self.check(a_in, a_out, a_mask) * a_mask.reshape((-1,1))).mean(axis=0)
        assert_greater(a_test[0], a_test[1] * 1.25)

    def test_FourLabels(self):
        # Only one sample has weight 1.0, the others have weight 0.0. Check probabilities!
        chosen = random.randint(0,15)
        a_in = numpy.random.uniform(-1.0, +1.0, (16,8))
        a_out = numpy.random.randint(2, size=(16,4))
        a_mask = numpy.zeros((16,), dtype=numpy.float32)
        a_mask[chosen] = 1.0

        a_test = self.check(a_in, a_out, a_mask, act="Sigmoid")
        print(a_out[chosen])

        for i in range(a_out.shape[1]):
            assert_equals(a_test[i][chosen,0] < a_test[i][chosen,1], a_out[chosen,i])
