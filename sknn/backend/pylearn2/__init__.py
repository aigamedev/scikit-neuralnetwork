# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

from ...nn import ansi


import warnings
warnings.warn(ansi.YELLOW + """\n
The PyLearn2 backend is deprecated; the next release will switch to Lasagne by default.

Test the change using the following at the top of your script:
> from sknn.backend import lasagne
""" + ansi.ENDC, category=UserWarning)


from ... import backend
from .mlp import MultiLayerPerceptronBackend
from .ae import AutoEncoderBackend

# Register this implementation as the MLP backend.
backend.MultiLayerPerceptronBackend = MultiLayerPerceptronBackend
backend.AutoEncoderBackend = AutoEncoderBackend
backend.name = 'pylearn2'