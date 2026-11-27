
from sklearn.base import BaseEstimator as SklearnBaseEstimator


class BaseEstimator(SklearnBaseEstimator):
    # http://msmbuilder.org/development/apipatterns.html

    def __dir__(self):
        # scikit-learn's __dir__ keeps the attributes for which hasattr() is
        # true. Fitted-only properties of MSMBuilder estimators (timescales_,
        # score_, ...) raise TypeError rather than AttributeError before fit,
        # which made dir() -- and with it IPython display and tab completion
        # -- fail on unfitted estimators. Leave those attributes out instead.
        names = []
        for name in object.__dir__(self):
            try:
                if hasattr(self, name):
                    names.append(name)
            except Exception:  # noqa: BLE001 - an unfitted property
                pass
        return names

    def summarize(self):
        """Return some diagnostic summary statistics about this Markov model"""
        return 'NotImplemented'
