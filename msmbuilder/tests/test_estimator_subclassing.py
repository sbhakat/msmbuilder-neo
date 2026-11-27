
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
