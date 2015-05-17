# -*- coding: utf-8 -*-
# This is a self-contained wrapper for PyLearn2, which contains only the relevant
# modules for scikit-neuralnetwork's implementation.  This allows us to package
# ``pylearn2`` along with ``sknn`` in PIP packages, but fallback to the global
# installation if it's available.
from __future__ import (absolute_import, unicode_literals, print_function)

import os
import sys


# Setup self-contained distributions for PIP packaged PyLearn2.
try:
    import pylearn2
except ImportError:
    pwd = os.path.dirname(__file__)
    if os.path.exists(os.path.join(pwd, 'pylearn2')):
        sys.path.insert(0, pwd)


# This wrapper module can only contain 'leaf' modules.
from pylearn2 import space
from pylearn2 import datasets
from pylearn2 import termination_criteria
from pylearn2.models import (mlp, maxout)
from pylearn2.training_algorithms import (sgd, learning_rule)
from pylearn2.costs import mlp as costs
from pylearn2.costs.mlp import dropout
from pylearn2.costs.cost import SumOfCosts as SumOfCosts
