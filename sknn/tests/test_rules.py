import unittest
from nose.tools import (assert_is_not_none, assert_false, assert_raises, assert_equal)

import numpy as np


from sknn.mlp import MultiLayerPerceptronRegressor as MLPR


class TestLearningRules(unittest.TestCase):

    def test_default(self):
        self._run(MLPR(layers=[("Linear",)],
                       learning_rule='default'))

    def test_momentum(self):
        self._run(MLPR(layers=[("Linear",)],
                       learning_rule='momentum'))

    def test_rmsprop(self):
        self._run(MLPR(layers=[("Linear",)],
                       learning_rule='rmsprop'))

    def test_rmsprop(self):
        self._run(MLPR(layers=[("Linear",)],
                       dropout=True)

    def test_unknown(self):
        assert_raises(NotImplementedError, MLPR,
                      layers=[], learning_rule='unknown')

    def _run(self, nn):
        a_in, a_out = np.zeros((8,16)), np.zeros((8,4))
        nn.fit(a_in, a_out)
        a_test = nn.predict(a_in)
        assert_equal(type(a_out), type(a_test))
