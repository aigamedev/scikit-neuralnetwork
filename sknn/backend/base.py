# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)


class BaseBackend(object):
    """Base class that all backends should inherit from.  This provides
    helper functions to make it easy to access all configuration from
    the user.
    """

    def __init__(self, spec):
        self.spec = spec
    
    def __getattr__(self, key):
        return getattr(self.spec, key)

    def __setattr__(self, key, value):
        if key != 'spec' and hasattr(self.spec, key):
            self.spec.__setattr__(key, value)
        else:
            super(BaseBackend, self).__setattr__(key, value)
