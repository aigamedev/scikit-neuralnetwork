# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

__author__ = 'ssamot, alexjc'
__version__ = '0.1'


import os
import sys
import logging


class TheanoConfigurator(object):

    def __init__(self):
        self.configured = False

    def configure(self, flags):
        if self.configured or 'theano' in sys.modules:
            return

        os.environ.setdefault('THEANO_FLAGS', flags+',print_active_device=False')
        print('flags', flags)
        cuda = logging.getLogger('theano.sandbox.cuda')
        cuda.setLevel(logging.CRITICAL)
        import theano
        cuda.setLevel(logging.WARNING)

        self.configured = True

    def __getattr__(self, name):
        flags = ''
        if name.endswith('32'):
            flags = ',floatX=float32'
        if name.endswith('64'):
            flags = ',floatX=float32'

        if name.startswith('cpu'):
            return self.configure('device=cpu'+flags)
        if name.startswith('gpu'):
            return self.configure('device=gpu'+flags)

        return getattr(sys.modules['sknn'], name)


sys.modules['sknn.backend'] = TheanoConfigurator()
