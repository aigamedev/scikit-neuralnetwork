# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)


# Once a submodule has been imported, its name will be stored here.
name = None

# This placeholder multi-layer perceptron class is replaced on importing
# a submodule from this backend module, e.g. from PyLearn2's implementation.
class MultiLayerPerceptronBackend(object):
    def __init__(self):
        raise NotImplementedError("No backend sub-module imported.")

# This placeholder auto-encoder class is replaced on importing a submodule
# from this backend module, e.g. from PyLearn2's implementation.
class AutoEncoderBackend(object):
    def __init__(self):
        raise NotImplementedError("No backend sub-module imported.")
