
from sklearn.base import BaseEstimator as SklearnBaseEstimator


class BaseEstimator(SklearnBaseEstimator):
    # http://msmbuilder.org/development/apipatterns.html

    def summarize(self):
        """Return some diagnostic summary statistics about this Markov model"""
        return 'NotImplemented'
