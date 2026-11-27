# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors: Matthew Harrigan <matthew.harrigan@outlook.com>
# Copyright (c) 2015, Stanford University
# All rights reserved.


import getpass
import glob
import os
import re
import socket
import sys
import warnings
from collections.abc import Sequence
from datetime import datetime
from os.path import join, exists, expanduser

import h5py
import mdtraj as md
import numpy as np
from mdtraj.core.trajectory import _parse_topology

from . import version

__all__ = ['dataset']

warnings.warn("This module might be deprecated in favor of msmbuilder.io",
              PendingDeprecationWarning)


def dataset(path, mode='r', fmt=None, verbose=False, **kwargs):
    """Open a dataset object

    MSMBuilder supports several dataset 'formats' for storing
    lists of sequences on disk.

    This function can also be used as a context manager.

    Parameters
    ----------
    path : str
        The path to the dataset on the filesystem
    mode : {'r', 'w', 'a'}
        Open a dataset for reading, writing, or appending. Note that
        some formats only support a subset of these modes.
    fmt : {'dir-npy', 'hdf5', 'mdtraj'}
        The format of the data on disk

        ``dir-npy``
            A directory of binary numpy files, one file per sequence

        ``hdf5``
            A single HDF5 file with each sequence as an array node,
            written with h5py. Files written by earlier MSMBuilder versions
            (with PyTables) open unchanged, and files written now open in
            them. Pass ``compression='gzip'`` (or 'lzf') to compress new
            arrays.

        ``mdtraj``
            A read-only set of trajectory files that can be loaded
            with mdtraj

    verbose : bool
        Whether to print information about the dataset

    """

    if mode == 'r' and fmt is None:
        fmt = _guess_format(path)
    elif mode in 'wa' and fmt is None:
        raise ValueError('mode="%s", but no fmt. fmt=%s' % (mode, fmt))

    if fmt == 'dir-npy':
        return NumpyDirDataset(path, mode=mode, verbose=verbose)
    elif fmt == 'mdtraj':
        return MDTrajDataset(path, mode=mode, verbose=verbose, **kwargs)
    elif fmt == 'hdf5':
        return HDF5Dataset(path, mode=mode, verbose=verbose, **kwargs)
    elif fmt.endswith("-union"):
        raise ValueError("union datasets have been removed. "
                         "Please use msmbuilder.featurizer.FeatureUnion")
    else:
        raise NotImplementedError("Unknown format fmt='%s'" % fmt)


def _guess_format(path):
    """Guess the format of a dataset based on its filename / filenames.
    """
    if os.path.isdir(path):
        return 'dir-npy'

    if path.endswith('.h5') or path.endswith('.hdf5'):
        # TODO: Check for mdtraj .h5 file
        return 'hdf5'

    # TODO: What about a list of trajectories, e.g. from command line nargs='+'
    return 'mdtraj'


class _BaseDataset(Sequence):
    _PROVENANCE_TEMPLATE = '''MSMBuilder Dataset:
  MSMBuilder:\t{version}
  Command:\t{cmdline}
  Path:\t\t{path}
  Username:\t{user}
  Hostname:\t{hostname}
  Date:\t\t{date}
  Comments:\t\t{comments}
'''
    _PREV_TEMPLATE = '''
== Derived from ==
{previous}
'''

    def __init__(self, path, mode='r', verbose=False):
        self.path = path
        self.mode = mode
        self.verbose = verbose

        if mode not in ('r', 'w', 'a'):
            raise ValueError('mode must be one of "r", "w", "a"')
        if mode in 'wa':
            if mode == 'w' and exists(path):
                raise ValueError('File exists: %s' % path)
            try:
                os.makedirs(path)
            except OSError:
                pass
            self._write_provenance()

    def create_derived(self, out_path, comments='', fmt=None):
        if fmt is None:
            out_dataset = self.__class__(out_path, mode='w',
                                         verbose=self.verbose)
        else:
            out_dataset = dataset(out_path, mode='w', verbose=self.verbose,
                                  fmt=fmt)
        out_dataset._write_provenance(previous=self.provenance,
                                      comments=comments)
        return out_dataset

    def apply(self, fn):
        for key in self.keys():
            yield fn(self.get(key))

    def _build_provenance(self, previous=None, comments=''):
        val = self._PROVENANCE_TEMPLATE.format(
            version=version.full_version,
            cmdline=' '.join(sys.argv),
            user=getpass.getuser(),
            hostname=socket.gethostname(),
            path=self.path,
            comments=comments,
            date=datetime.now().strftime("%B %d, %Y %I:%M %p"))
        if previous:
            val += self._PREV_TEMPLATE.format(previous=previous)
        return val

    def _lazy(self, key):
        """An array-like for sequence ``key`` that reads only the slices
        asked for (used by :meth:`to_dask`)."""
        raise NotImplementedError('%s datasets do not support lazy loading'
                                  % type(self).__name__)

    def to_dask(self, chunks='auto'):
        """The sequences as a list of lazily loaded Dask arrays.

        Nothing is read until the arrays are computed, and then only the
        chunks needed, so datasets larger than memory can be reduced,
        sliced or fed to estimators with ``partial_fit`` chunk by chunk.

        Parameters
        ----------
        chunks : str, int, tuple, default='auto'
            Dask chunk specification for each array (see
            ``dask.array.from_array``).

        Returns
        -------
        arrays : list of dask.array.Array
            One per sequence, in ``keys()`` order. For HDF5 datasets, keep
            the dataset open while computing them.

        Notes
        -----
        Needs Dask: ``pip install msmbuilder-neo[dask]``.
        """
        try:
            import dask.array as da
        except ImportError as e:
            raise ImportError('to_dask needs Dask. Install it with "pip install dask" '
                              'or "pip install msmbuilder-neo[dask]".') from e
        return [da.from_array(self._lazy(key), chunks=chunks, lock=True,
                              name='msmbuilder-%s-%d' % (os.path.abspath(str(self.path)), key))
                for key in self.keys()]

    def fit_with(self, estimator):
        """Call the fit method of the estimator on this dataset

        Parameters
        ----------
        estimator : BaseEstimator
            estimator.fit will be called on this dataset.

        Returns
        -------
        estimator
            The fit estimator.
        """
        estimator.fit(self)
        return estimator

    def transform_with(self, estimator, out_ds, fmt=None):
        """Call the partial_transform method of the estimator on this dataset

        Parameters
        ----------
        estimator : object with ``partial_fit`` method
            This object will be used to transform this dataset into a new
            dataset. The estimator should be fitted prior to calling
            this method.
        out_ds : str or Dataset
            This dataset will be transformed and saved into out_ds. If
            out_ds is a path, a new dataset will be created at that path.
        fmt : str
            The type of dataset to create if out_ds is a string.

        Returns
        -------
        out_ds : Dataset
            The tranformed dataset.
        """
        if isinstance(out_ds, str):
            out_ds = self.create_derived(out_ds, fmt=fmt)
        elif isinstance(out_ds, _BaseDataset):
            err = "Dataset must be opened in write mode."
            assert out_ds.mode in ('w', 'a'), err
        else:
            err = "Please specify a dataset path or an existing dataset."
            raise ValueError(err)
        for key in self.keys():
            out_ds[key] = estimator.partial_transform(self[key])
        return out_ds

    def fit_transform_with(self, estimator, out_ds, fmt=None):
        """Create a new dataset with the given estimator.

        The estimator will be fit by this dataset, and then each trajectory
        will be transformed by the estimator.

        Parameters
        ----------
        estimator : BaseEstimator
            This object will be fit and used to transform this dataset
            into a new dataset.
        out_ds : str or Dataset
            This dataset will be transformed and saved into out_ds. If
            out_ds is a path, a new dataset will be created at that path.
        fmt : str
            The type of dataset to create if out_ds is a string.

        Returns
        -------
        out_ds : Dataset
            The transformed dataset.

        Examples
        --------
        diheds = dataset("diheds")
        tica = diheds.fit_transform_with(tICA(), 'tica')
        kmeans = tica.fit_transform_with(KMeans(), 'kmeans')
        msm = kmeans.fit_with(MarkovStateModel())
        """
        self.fit_with(estimator)
        return self.transform_with(estimator, out_ds, fmt=fmt)

    @property
    def provenance(self):
        raise NotImplementedError('implemented in subclass')

    def _write_provenance(self, previous=None, comments=''):
        raise NotImplementedError('implemented in subclass')

    def __len__(self):
        return sum(1 for xx in self.keys())

    def __getitem__(self, i):
        if isinstance(i, slice):
            err = "Please index datasets with explicit indices or ds[:]."
            assert i.start is None and i.step is None and i.stop is None, err
            return [self.get(i) for i in self.keys()]
        return self.get(i)

    def __setitem__(self, i, x):
        return self.set(i, x)

    def __iter__(self):
        for key in self.keys():
            yield self.get(key)

    def keys(self):
        # keys()[i], get(i) and set(i, x) should all follow
        # the same ordering convention for the indices / items.
        raise NotImplementedError('implemeneted in subclass')

    def items(self):
        for key in self.keys():
            yield (key, self.get(key))

    def get(self, i):
        raise NotImplementedError('implemeneted in subclass')

    def set(self, i, x):
        raise NotImplementedError('implemeneted in subclass')

    def close(self):
        pass

    def flush(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


class NumpyDirDataset(_BaseDataset):
    """MSMBuilder dataset container

    Parameters
    ----------
    path : str
    mode : {'r', 'w', 'a'}
        Read, write, or append. If mode is set to 'a' or 'w',
        duplicate keys will be overwritten.

    Examples
    --------
    for X in Dataset('path/to/dataset'):
        print X
    """

    _ITEM_FORMAT = '%08d.npy'
    _ITEM_RE = re.compile(r'(\d{8}).npy')
    _PROVENANCE_FILE = 'PROVENANCE.txt'

    def get(self, i, mmap=False):
        mmap_mode = 'r' if mmap else None

        filename = join(self.path, self._ITEM_FORMAT % i)
        if self.verbose:
            print('[NumpydirDataset] loading %s' % filename)
        try:
            return np.load(filename, mmap_mode)
        except OSError as e:
            raise IndexError(e)

    def _lazy(self, key):
        return self.get(key, mmap=True)

    def set(self, i, x):
        if self.mode not in 'wa':
            raise OSError('Dataset not opened for writing')
        filename = join(self.path, self._ITEM_FORMAT % i)
        if self.verbose:
            print('[NumpydirDataset] saving %s' % filename)
        return np.save(filename, x)

    def keys(self):
        for fn in sorted(os.listdir(os.path.expanduser(self.path)),
                         key=_keynat):
            match = self._ITEM_RE.match(fn)
            if match:
                yield int(match.group(1))

    @property
    def provenance(self):
        try:
            with open(join(self.path, self._PROVENANCE_FILE)) as f:
                return f.read()
        except OSError:
            return 'No available provenance'

    def _write_provenance(self, previous=None, comments=''):
        with open(join(self.path, self._PROVENANCE_FILE), 'w') as f:
            p = self._build_provenance(previous=previous, comments=comments)
            f.write(p)


class HDF5Dataset(_BaseDataset):
    """Sequences stored as arrays ``arr_<key>`` in one HDF5 file (h5py).

    The layout is the one earlier MSMBuilder versions wrote with PyTables:
    one array per sequence under the root group, and the provenance text in
    the root attribute ``provenance``. Existing files open unchanged, and
    files written here open in earlier versions.

    Parameters
    ----------
    path : str
    mode : {'r', 'w', 'a'}
        Read, create (the file must not exist), or append (create if
        missing). In 'w' and 'a' modes existing keys are overwritten.
    verbose : bool
    compression : str, optional
        h5py compression filter for new arrays, e.g. 'gzip' or 'lzf'.
    """
    _ITEM_FORMAT = 'arr_%d'
    _ITEM_RE = re.compile(r'arr_(\d+)$')

    def __init__(self, path, mode='r', verbose=False, compression=None):
        if mode not in ('r', 'w', 'a'):
            raise ValueError('mode must be one of "r", "w", "a"')
        if mode == 'w' and exists(path):
            raise ValueError('File exists: %s' % path)
        self.path = path
        self.mode = mode
        self.verbose = verbose
        self.compression = compression
        self._open()
        if mode == 'w' or (mode == 'a' and 'provenance' not in self._handle.attrs):
            self._write_provenance()

    def _open(self):
        h5mode = {'r': 'r', 'w': 'w-', 'a': 'a'}[self.mode]
        self._handle = h5py.File(self.path, h5mode)

    def __getstate__(self):
        # h5py file handles cannot be pickled; reopen on unpickling
        return {'path': self.path, 'mode': self.mode, 'verbose': self.verbose,
                'compression': self.compression}

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.mode == 'w':
            self.mode = 'a'     # the file exists now
        self._open()

    def get(self, i, mmap=False):
        try:
            node = self._handle[self._ITEM_FORMAT % i]
        except KeyError:
            raise IndexError('no sequence %r in %s' % (i, self.path)) from None
        if self.verbose:
            print('[HDF5Dataset] loading %s/%s' % (self.path, node.name))
        return node[()]

    def _lazy(self, key):
        return self._handle[self._ITEM_FORMAT % key]

    def keys(self):
        names = [n for n in self._handle.keys() if self._ITEM_RE.match(n)]
        for name in sorted(names, key=_keynat):
            yield int(self._ITEM_RE.match(name).group(1))

    def set(self, i, x):
        if self.mode not in ('w', 'a'):
            raise OSError('Dataset not opened for writing')
        name = self._ITEM_FORMAT % i
        if name in self._handle:
            del self._handle[name]
        x = np.asarray(x)
        kwargs = {}
        if self.compression is not None and x.size > 0:
            kwargs = dict(compression=self.compression, chunks=True)
        self._handle.create_dataset(name, data=x, **kwargs)

    @property
    def provenance(self):
        value = self._handle.attrs.get('provenance')
        if value is None:
            return 'No available provenance'
        if isinstance(value, bytes):
            value = value.decode('utf-8', 'replace')
        return str(value)

    def _write_provenance(self, previous=None, comments=''):
        p = self._build_provenance(previous=previous, comments=comments)
        self._handle.attrs['provenance'] = p

    def close(self):
        handle = getattr(self, '_handle', None)
        if handle is not None and handle.id.valid:
            handle.close()

    def flush(self):
        self._handle.flush()

    def __del__(self):
        self.close()


class MDTrajDataset(_BaseDataset):
    _PROVENANCE_TEMPLATE = '''MDTraj dataset:
  path:\t\t{path}
  topology:\t{topology}
  stride:\t{stride}
  atom_indices\t{atom_indices}
'''

    def __init__(self, path, mode='r', topology=None, stride=1,
                 atom_indices=None, verbose=False):
        if mode != 'r':
            raise ValueError('mode must be "r"')
        self.path = path
        self.topology = topology
        self.stride = stride
        self.atom_indices = atom_indices
        self.verbose = verbose

        if isinstance(path, list):
            self.glob_matches = [expanduser(fn) for fn in path]
        else:
            self.glob_matches = sorted(glob.glob(expanduser(path)),
                                       key=_keynat)

        if topology is None:
            self._topology = None
        else:
            self._topology = _parse_topology(os.path.expanduser(topology))

    def get(self, i):
        if self.verbose:
            print('[MDTraj dataset] loading %s' % self.filename(i))

        if self._topology is None:
            t = md.load(self.filename(i), stride=self.stride,
                        atom_indices=self.atom_indices)
        else:
            t = md.load(self.filename(i), stride=self.stride,
                        atom_indices=self.atom_indices, top=self._topology)
        return t

    def filename(self, i):
        return self.glob_matches[i]

    def iterload(self, i, chunk):
        if self.verbose:
            print('[MDTraj dataset] iterloading %s' % self.filename(i))

        if self._topology is None:
            return md.iterload(
                self.filename(i), chunk=chunk, stride=self.stride,
                atom_indices=self.atom_indices)
        else:
            return md.iterload(
                self.filename(i), chunk=chunk, stride=self.stride,
                atom_indices=self.atom_indices, top=self._topology)

    def keys(self):
        return iter(range(len(self.glob_matches)))

    @property
    def provenance(self):
        return self._PROVENANCE_TEMPLATE.format(
            path=self.path, topology=self.topology,
            atom_indices=self.atom_indices, stride=self.stride)


def _dim_match(arr):
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    return arr


def _keynat(string):
    """A natural sort helper function for sort() and sorted()
    without using regular expression.
    """
    r = []
    for c in string:
        if c.isdigit():
            if r and isinstance(r[-1], int):
                r[-1] = r[-1] * 10 + int(c)
            else:
                r.append(int(c))
        else:
            r.append(9 + ord(c))
    return r
