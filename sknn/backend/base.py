# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)


class BackendBase(object):

    def __init__(self, spec):
        self.spec = spec
    
    def __getattr__(self, key):
        return getattr(self.spec, key)
