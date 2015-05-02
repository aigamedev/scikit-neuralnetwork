# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

from pylearn2.datasets.dataset import Dataset

from pylearn2.utils.iteration import (SequentialSubsetIterator,
                                      FiniteDatasetIterator,
                                      resolve_iterator_class)

import functools

from pylearn2.space import CompositeSpace, VectorSpace
from pylearn2.utils import safe_zip

import theano
floatX = theano.config.floatX


class SparseDesignMatrix(Dataset):
    """
    SparseExpanderDataset takes a numpy/scipy sparse matrix and calls .todense()
    as the batches are passed out of the iterator.
    """

    def __init__(self, X, y):
        self.X = X.astype(floatX)
        self.y = y.astype(floatX)

        self.data_n_rows = self.X.shape[0]
        self.num_examples = self.data_n_rows
        self.fancy = False
        self.stochastic = False
        X_space = VectorSpace(dim=self.X.shape[1])
        X_source = 'features'

        if self.y.ndim == 1:
            dim = 1
        else:
            dim = self.y.shape[-1]
        y_space = VectorSpace(dim=dim)
        y_source = 'targets'

        space = CompositeSpace((X_space, y_space))
        source = (X_source, y_source)

        self.data_specs = (space, source)
        self.X_space = X_space
        self._iter_data_specs = (self.X_space, 'features')

    def get_num_examples(self):
        return self.num_examples

    def get_design_matrix(self):
        return self.X

    def get_batch_design(self, batch_size, include_labels=False):
        """
        method inherited from Dataset
        """
        self.iterator(mode='sequential', batch_size=batch_size)
        return self.next()

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.
        """
        return self.data_specs

    def get_data(self):
      """
      Returns
      -------
      data : numpy matrix or 2-tuple of matrices
          Returns all the data, as it is internally stored.
          The definition and format of these data are described in
          `self.get_data_specs()`.
      """
      if self.y is None:
          return self.X
      else:
          return (self.X, self.y)

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
      topo=None, targets=None, rng=None, data_specs=None,
      return_tuple=False):
        """
        method inherited from Dataset
        """
        self.mode = mode
        self.batch_size = batch_size
        self._targets = targets
        self._return_tuple = return_tuple
        if data_specs is None:
                data_specs = self._iter_data_specs

        # TODO: If there is a view_converter, we have to use it to convert
        # the stored data for "features" into one that the iterator can return.

        def convertion_function(x):
            print(type(x), x.dtype, x.shape)
            return x.todense().astype(floatX)

        self.conv_fn = convertion_function

        space, source = data_specs
        if isinstance(space, CompositeSpace):
            sub_spaces = space.components
            sub_sources = source
        else:
            sub_spaces = (space,)
            sub_sources = (source,)

        convert = []
        for sp, src in safe_zip(sub_spaces, sub_sources):
            if src == 'features' or 'targets':
                conv_fn = self.conv_fn
            else:
                conv_fn = None

            convert.append(conv_fn)

        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        subset_iterator = mode(self.X.shape[0], batch_size, num_batches, rng)
        return FiniteDatasetIterator(self,
                                     subset_iterator,
                                     data_specs=data_specs,
                                     return_tuple=return_tuple,
                                     convert=convert)

    def __iter__(self):
        return self

    def next(self):
        indx = self.subset_iterator.next()
        try:
            rval = self.X[indx].todense()
        except IndexError:
            # the ind of minibatch goes beyond the boundary
            import ipdb; ipdb.set_trace()
        rval = tuple(rval)
        if not self._return_tuple and len(rval) == 1:
            rval, = rval
        return rval
