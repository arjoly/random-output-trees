'''
This module provides general purpose meta-transformer.

'''

# Authors: Arnaud Joly <arnaud.v.joly@gmail.com>
#
# License: BSD 3 clause

from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import TransformerMixin
from sklearn.utils import check_random_state


class FixedStateTransformer(BaseEstimator, TransformerMixin):
    """Fixe the random_state of the transformer

    This meta-transformer is usefull when you want to fix the random_state
    of a transformer, which is modified by some meta-estimator.

    Parameters
    ----------
    transformer : scikit-learn transformer

    random_seed : int, RandomState instance, optional (default=0)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;

    Attributes
    ----------
    transformer_ : transformer
        A clone of the fitted transformer

    """
    def __init__(self, transformer, random_seed=0):
        self.transformer = transformer
        self.random_seed = random_seed

        self.transformer_ = None

    @property
    def random_state(self):
        return self.random_seed

    def fit(self, X, y=None):
        """Fit estimator.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Input data used to build forests.

        Returns
        -------
        self : object
            Returns self.
        """
        random_state = check_random_state(self.random_seed)
        self.transformer_ = clone(self.transformer)

        try:
            self.transformer_.set_params(random_state=random_state)
        except ValueError:
            pass

        try:
            self.transformer_.fit(X, y)
        except TypeError:
            self.transformer_.fit(X)

        return self

    def transform(self, X):
        """Transform dataset.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Input data to be transformed.

        Returns
        -------
        X_transformed: sparse matrix, shape=(n_samples, n_out)
            Transformed dataset.
        """
        return self.transformer_.transform(X)
