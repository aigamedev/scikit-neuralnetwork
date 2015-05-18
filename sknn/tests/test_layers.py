import unittest
from nose.tools import (assert_equal, assert_raises, assert_in, assert_not_in)

from sknn.mlp import Regressor as MLPR
from sknn.mlp import Layer as L


class TestNestedParameters(unittest.TestCase):

    def test_GetParamsIncludesLayers(self):
        nn = MLPR(layers=[L("Linear", units=123)])
        p = nn.get_params()
        assert_in('output', p)

    def test_GetParamsMissingLayer(self):
        nn = MLPR(layers=[L("Linear", units=123)])
        p = nn.get_params()
        assert_not_in('hidden0', p)

    def test_SetParamsDoubleUnderscore(self):
        nn = MLPR(layers=[L("Linear", units=123)])
        nn.set_params(output__units=456)
        assert_equal(nn.layers[0].units, 456)

    def test_SetParamsValueError(self):
        nn = MLPR(layers=[L("Linear")])
        assert_raises(ValueError, nn.set_params, output__range=1.0)
