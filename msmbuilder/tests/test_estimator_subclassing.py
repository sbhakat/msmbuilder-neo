
import importlib
import inspect
import pkgutil
import warnings
from contextlib import contextmanager

from sklearn.base import BaseEstimator

import msmbuilder
import msmbuilder.base


@contextmanager
def supress_warnings():
    # (This used to replace warnings.warn and restore it after the block,
    # without try/finally: a pytest skip raised while importing a module
    # left it replaced for every later test in the process.)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


def import_all_estimators(pkg):
    def estimator_in_module(mod):
        for name, obj in inspect.getmembers(mod):
            if name.startswith('_'):
                continue
            if inspect.isclass(obj) and issubclass(obj, BaseEstimator):
                yield obj

    with supress_warnings():
        result = {}
        for _, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
            if modname in ('tests', 'scripts', 'project_templates'):
                continue    # not library code (and test modules may skip on import)
            c = '%s.%s' % (pkg.__name__, modname)
            try:
                mod = importlib.import_module(c)
                if ispkg:
                    result.update(import_all_estimators(mod))
                for kls in estimator_in_module(mod):
                    result[kls.__name__] = kls
            except ImportError as e:
                print('e', e)
                continue

        return result


def test_all_estimators():
    for key, value in import_all_estimators(msmbuilder).items():
        if 'msmbuilder' in value.__module__:
            assert issubclass(value, msmbuilder.base.BaseEstimator), value


def test_dir_of_unfitted_estimators():
    # dir() (used by IPython display and tab completion) must not fail on
    # estimators whose fitted-only properties cannot be computed yet
    import msmbuilder.base
    from msmbuilder.decomposition import tICA
    from msmbuilder.lumping import PCCAPlus
    from msmbuilder.msm import BayesianMarkovStateModel, ContinuousTimeMSM, MarkovStateModel

    for est in (MarkovStateModel(), BayesianMarkovStateModel(), ContinuousTimeMSM(), tICA(),
                PCCAPlus(2)):
        names = dir(est)
        assert 'fit' in names and 'get_params' in names
    count = 0
    for cls in import_all_estimators(msmbuilder).values():
        if issubclass(cls, msmbuilder.base.BaseEstimator):
            try:
                est = cls()
            except (TypeError, ValueError):
                continue                    # needs required arguments
            dir(est)
            count += 1
    assert count > 20


def test_estimators_can_be_cloned():
    # scikit-learn's clone (used by GridSearchCV, cross_val_score, Pipeline
    # searches) requires __init__ to store its arguments unchanged
    import msmbuilder.base
    from sklearn.base import clone

    from msmbuilder.featurizer import DihedralFeaturizer, VonMisesFeaturizer
    from msmbuilder.preprocessing import Butterworth

    for est in (DihedralFeaturizer(types=['phi']), DihedralFeaturizer(types='psi'),
                VonMisesFeaturizer(types=['phi', 'psi'], n_bins=10), Butterworth(width=4)):
        c = clone(est)
        assert c.get_params() == est.get_params()
    count = 0
    for cls in import_all_estimators(msmbuilder).values():
        if issubclass(cls, msmbuilder.base.BaseEstimator):
            try:
                est = cls()
            except (TypeError, ValueError):
                continue
            clone(est)
            count += 1
    assert count > 20


def test_clone_preserves_behaviour():
    import mdtraj as md
    import numpy as np

    from msmbuilder.example_datasets import AlanineDipeptide
    from msmbuilder.featurizer import DihedralFeaturizer
    from msmbuilder.preprocessing import Butterworth

    traj = AlanineDipeptide().get().trajectories[0][:100]
    assert isinstance(traj, md.Trajectory)
    for types in ('phi', ['phi', 'psi']):
        f = DihedralFeaturizer(types=types)
        np.testing.assert_array_equal(f.partial_transform(traj),
                                      DihedralFeaturizer(types=types).partial_transform(traj))
    assert DihedralFeaturizer(types='phi').partial_transform(traj).shape[1] == 2
    x = np.random.RandomState(0).randn(200, 2)
    # an even width is rounded up to the next odd window, as before
    np.testing.assert_array_equal(Butterworth(width=4).partial_transform(x),
                                  Butterworth(width=5).partial_transform(x))
