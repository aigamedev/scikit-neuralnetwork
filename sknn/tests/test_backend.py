import unittest
from nose.tools import (assert_in, assert_equal)

import os
import sys

import sknn


class TestBackendPseudoModule(unittest.TestCase):

    def setUp(self):
        if 'THEANO_FLAGS' in os.environ:
            del os.environ['THEANO_FLAGS']
        for name in sys.modules.keys():
            if name.startswith('theano'):
                del sys.modules[name]
        sys.modules['sknn.backend'].configured = False

    def test_TheanoWarning(self):
        pass

    def _check(self, flags):
        assert_in('THEANO_FLAGS', os.environ)
        variable = os.environ['THEANO_FLAGS']
        for f in flags:
            assert_in(f, variable)

    def test_FlagsGPU32(self):
        from sknn.backend import gpu32
        self._check(['floatX=float32','device=gpu'])

    def test_FlagsCPU32(self):
        from sknn.backend import cpu32
        self._check(['floatX=float32','device=cpu'])

    def test_FlagsGPU64(self):
        from sknn.backend import gpu64
        self._check(['floatX=float64','device=gpu'])

    def test_FlagsCPU64(self):
        from sknn.backend import cpu64
        self._check(['floatX=float64','device=cpu'])
