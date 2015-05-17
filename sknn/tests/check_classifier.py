import unittest
from nose.tools import (assert_equals, assert_true)

from hypothesis import given
import hypothesis.strategies as S

from sknn.mlp import Classifier

ClassifierStrategy = S.builds(
    Classifier,
    S.lists(None, max_size=0))


class TestEncoding(unittest.TestCase):
    @given(ClassifierStrategy)
    def test_decode_inverts_encode(self, cls):
        assert_true(isinstance(cls, Classifier))
        assert_equals(cls.layers, [])

if __name__ == '__main__':
    unittest.main()
