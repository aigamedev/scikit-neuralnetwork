import unittest
from nose.tools import (assert_in, assert_raises, assert_equals)

import collections
import numpy
from sknn.mlp import MultiLayerPerceptron as MLP, Layer as L

import sknn.mlp


class TestSingleCallback(unittest.TestCase):

    def setUp(self):
        self.data = collections.defaultdict(list)

    def _callback(self, event, **variables):
        self.data[event].append(variables)

    def test_TrainingCallbacks(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn = MLP(layers=[L("Linear")], n_iter=4, callback=self._callback)
        nn._fit(a_in, a_out)
        assert_equals(len(self.data['on_train_start']), 1)
        assert_equals(len(self.data['on_train_finish']), 1)

    def test_EpochCallbacks(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn = MLP(layers=[L("Linear")], n_iter=4, callback=self._callback)
        nn._fit(a_in, a_out)
        assert_equals(len(self.data['on_epoch_start']), 4)
        assert_equals(len(self.data['on_epoch_finish']), 4)

    def test_BatchCallbacks(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn = MLP(layers=[L("Linear")], n_iter=1, batch_size=4, callback=self._callback)
        nn._fit(a_in, a_out)
        assert_equals(len(self.data['on_batch_start']), 2)
        assert_equals(len(self.data['on_batch_finish']), 2)


class TestSpecificCallback(unittest.TestCase):

    def setUp(self):
        self.data = []

    def _callback(self, **variables):
        self.data.append(variables)

    def test_TrainingCallback(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn = MLP(layers=[L("Linear")], n_iter=4, callback={'on_train_start': self._callback})
        nn._fit(a_in, a_out)
        assert_equals(len(self.data), 1)

    def test_EpochCallback(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn = MLP(layers=[L("Linear")], n_iter=4, callback={'on_epoch_start': self._callback})
        nn._fit(a_in, a_out)
        assert_equals(len(self.data), 4)

    def test_BatchCallbacks(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn = MLP(layers=[L("Linear")], n_iter=1, batch_size=4, callback={'on_batch_start': self._callback})
        nn._fit(a_in, a_out)
        assert_equals(len(self.data), 2)
