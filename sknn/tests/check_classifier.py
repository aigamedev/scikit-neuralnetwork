import unittest
from nose.tools import (assert_equals, assert_true)

from hypothesis import given
import hypothesis.strategies as S

from sknn.mlp import Classifier, Layer

LayerStrategy = S.builds(
    Layer,
    S.one_of(["Rectifier", "Sigmoid", "Tanh"])
)

ClassifierStrategy = S.builds(
    Classifier,
    S.lists(LayerStrategy, min_size=1, max_size=5))


class TestEncoding(unittest.TestCase):
    @given(ClassifierStrategy)
    def test_decode_inverts_encode(self, cls):
        assert_true(isinstance(cls, Classifier))
        assert_equals(cls.layers, [])

if __name__ == '__main__':
    unittest.main()
