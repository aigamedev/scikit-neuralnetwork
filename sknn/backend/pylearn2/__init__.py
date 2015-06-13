# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

from ... import backend
from .mlp import MultiLayerPerceptronBackend
from .ae import AutoEncoderBackend

# Register this implementation as the MLP backend.
backend.MultiLayerPerceptronBackend = MultiLayerPerceptronBackend
backend.AutoEncoderBackend = AutoEncoderBackend
