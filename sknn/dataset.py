__authors__ = "Nicholas Leonard"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Nicholas Leonard", "Yann Dauphin"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicholas Leonard"
__email__ = "leonardn@iro"
import functools

import warnings
import numpy as np
import scipy.sparse as spp
from pylearn2.utils.iteration import (
    FiniteDatasetIterator,
    FiniteDatasetIteratorPyTables,
    resolve_iterator_class,
    SequentialSubsetIterator
)
import copy
from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets import control
from theano import config
import theano
import gzip
floatX = theano.config.floatX

class SparseDesignMatrix(Dataset):
    """
    SparseDesignMatrix is by itself an iterator.
    """
    _default_seed = (17, 2, 946)
    def __init__(self, X, y=None, rng=None):

        self.X = X.astype(theano.config.floatX)
        self.y = y
        
        self.compress = False
        self.design_loc = None
        if rng is None:
            rng = np.random.RandomState(SparseDesignMatrix._default_seed)
        self.rng = rng
        
        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('sequential')
        self._iter_topo = False
        self._iter_targets = False

    def get_design_matrix(self):
        return self.X

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None):

        # TODO: Refactor
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)
        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if topo is None:
            topo = getattr(self, '_iter_topo', False)
        if targets is None:
            targets = getattr(self, '_iter_targets', False)
        if rng is None and mode.stochastic:
            rng = self.rng
        return FiniteDatasetIterator(self,
                                     mode(self.X.shape[0], batch_size,
                                     num_batches, rng),
                                     topo, targets)
    @property
    def num_examples(self):
        return self.X.shape[0]

    def get_batch_design(self, batch_size, include_labels=False):
        try:
            idx = self.rng.randint(self.X.shape[0] - batch_size + 1)
        except ValueError:
            if batch_size > self.X.shape[0]:
                raise ValueError("Requested "+str(batch_size)+" examples"
                    "from a dataset containing only "+str(self.X.shape[0]))
            raise
        rx = self.X[idx:idx + batch_size, :]
        if include_labels:
            if self.y is None:
                return rx, None
            ry = self.y[idx:idx + batch_size]
            return rx, ry
        return rx.astype(config.floatX)

    def get_batch_topo(self, batch_size):
        """
        method inherited from Dataset
        """
        raise NotImplementedError('Not implemented for sparse dataset')

        
    def get_targets(self):
        return self.y
        
    def has_targets(self):
         return self.y is not None
