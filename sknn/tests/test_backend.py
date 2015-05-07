import unittest
from nose.tools import (assert_in, assert_equal)

import io
import os
import sys
import logging

import sknn


class TestBackendPseudoModule(unittest.TestCase):

    def setUp(self):
        if 'THEANO_FLAGS' in os.environ:
            del os.environ['THEANO_FLAGS']
        
        import theano

        self.removed = {}
        for name in list(sys.modules.keys()):
            if name.startswith('theano'):
                self.removed[name] = sys.modules[name]
                del sys.modules[name]
        sys.modules['sknn.backend'].configured = False

        self.buf = io.StringIO()
        self.hnd = logging.StreamHandler(self.buf)
        logging.getLogger('sknn').addHandler(self.hnd)
        logging.getLogger().setLevel(logging.WARNING)

    def tearDown(self):
        for name, module in self.removed.items():
            sys.modules[name] = module
        logging.getLogger('sknn').removeHandler(self.hnd)

    def test_TheanoWarning(self):
        import theano
        from sknn.backend import cpu
        assert_equal('Theano was already imported and cannot be reconfigured.\n',
                     self.buf.getvalue())

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
