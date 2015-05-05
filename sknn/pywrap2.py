# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

import os
import sys


# Self-contained distributions on PIP package PyLearn2.
pwd = os.path.dirname(__file__)
if os.path.exists(os.path.join(pwd, 'pylearn2')):
    sys.path.insert(0, pwd)


# This wrapper module can only contain 'leaf' modules.
from pylearn2 import space
from pylearn2 import datasets
from pylearn2 import termination_criteria
from pylearn2.models import (mlp, maxout)
from pylearn2.training_algorithms import (sgd, learning_rule)
from pylearn2.costs.mlp import dropout
