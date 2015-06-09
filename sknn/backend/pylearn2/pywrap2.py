# -*- coding: utf-8 -*-
# This is a self-contained wrapper for PyLearn2, which contains only the relevant
# modules for scikit-neuralnetwork's implementation.  This allows us to package
# ``pylearn2`` along with ``sknn`` in PIP packages, but fallback to the global
# installation if it's available.
from __future__ import (absolute_import, unicode_literals, print_function)

import os
import sys
import logging


# Setup self-contained distributions for PIP packaged PyLearn2.
try:
    import pylearn2
except ImportError:
    pwd = os.path.dirname(__file__)
    if os.path.exists(os.path.join(pwd, 'pylearn2')):
        sys.path.insert(0, pwd)


# This wrapper module can only contain 'leaf' modules.
from pylearn2 import (space, datasets, blocks, corruption, utils, termination_criteria)
from pylearn2.datasets import (transformer_dataset, dataset)
from pylearn2.utils import (iteration)
from pylearn2.models import (mlp, maxout, autoencoder)
from pylearn2.training_algorithms import (sgd, learning_rule)
from pylearn2.costs import (mlp as mlp_cost, autoencoder as ae_costs, cost)
from pylearn2.costs.mlp import (dropout)


# Configure logging so we have full control of regular cases.
sgd.log.setLevel(logging.WARNING)
mlp.logger.setLevel(logging.WARNING)
