from __future__ import division

import numbers
from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.ensemble.base import BaseEnsemble
from sklearn.externals.six import with_metaclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_random_state
from sklearn.utils.validation import column_or_1d
from sklearn.utils.random import sample_without_replacement

from .._utils import check_array
from .._utils import check_X_y
from .._utils import has_fit_parameter


MAX_INT = np.iinfo(np.int32).max


def _generator_fitted_estimators(n_estimators, ensemble, X, y, sample_weight,
                                 seeds, verbose):
    """Private function used to build an iterator of estimators."""
    # Modified from sklearn.ensemble.bagging._parallel_build_estimators

    # Retrieve settings
    n_samples, n_features = X.shape
    max_samples = ensemble.max_samples
    max_features = ensemble.max_features

    if (not isinstance(max_samples, (numbers.Integral, np.integer)) and
            (0.0 < max_samples <= 1.0)):
        max_samples = int(max_samples * n_samples)

    if (not isinstance(max_features, (numbers.Integral, np.integer)) and
            (0.0 < max_features <= 1.0)):
        max_features = int(max_features * n_features)

    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.base_estimator_,
                                              "sample_weight")

    # Build estimators
    for i in range(n_estimators):
        if verbose > 1:
            print("building estimator %d of %d" % (i + 1, n_estimators))

        random_state = check_random_state(seeds[i])
        seed = check_random_state(random_state.randint(MAX_INT))
        estimator = ensemble._make_estimator(append=False)

        try:  # Not all estimator accept a random_state
            estimator.set_params(random_state=seed)
        except ValueError:
            pass

        # Draw features
        if bootstrap_features:
            features = random_state.randint(0, n_features, max_features)
        else:
            features = sample_without_replacement(n_features,
                                                  max_features,
                                                  random_state=random_state)

        # Draw samples, using sample weights, and then fit
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            if bootstrap:
                indices = random_state.randint(0, n_samples, max_samples)
                sample_counts = np.bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts

            else:
                not_indices = sample_without_replacement(
                    n_samples,
                    n_samples - max_samples,
                    random_state=random_state)

                curr_sample_weight[not_indices] = 0

            estimator.fit(X[:, features], y, sample_weight=curr_sample_weight)
            samples = curr_sample_weight > 0.

        # Draw samples, using a mask, and then fit
        else:
            if bootstrap:
                indices = random_state.randint(0, n_samples, max_samples)
            else:
                indices = sample_without_replacement(n_samples,
                                                     max_samples,
                                                     random_state=random_state)

            sample_counts = np.bincount(indices, minlength=n_samples)

            estimator.fit((X[indices])[:, features], y[indices])
            samples = sample_counts > 0.


        yield estimator, samples, features


class BaseLazyBagging(with_metaclass(ABCMeta, BaseEnsemble)):
    """Base class for Lazy Bagging meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 random_state=None,
                 verbose=0):
        super(BaseLazyBagging, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators)

        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.random_state = random_state
        self.verbose = verbose

        self._X = None
        self._y = None
        self._sample_weight = None

    def fit(self, X, y, sample_weight=None):
        """Build a lazy a bagging ensemble of estimators from the training set

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check inputs
        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], multi_output=True)
        self.n_features_ = X.shape[1]

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            column_or_1d(y, warn=True)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if self.n_estimators <= 0:
            raise ValueError("n_estimators should be greater than 0, got %s"
                             % self.n_estimators)

        if isinstance(self.max_samples, (numbers.Integral, np.integer)):
            max_samples = self.max_samples
        else:  # float
            max_samples = int(self.max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            max_features = int(self.max_features * self.n_features_)

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        # Retain X and y value
        self._X = X.copy()
        self._y = y.copy()
        self._sample_weight = sample_weight

        # Reproducibility with a common seed
        random_state = check_random_state(self.random_state)
        self.random_seed_ = random_state.randint(MAX_INT)

        if self.n_outputs_ == 1:
            self._y = y.ravel()

            if isinstance(self, ClassifierMixin):
                self.n_classes_ = self.n_classes_[0]
                self.classes_ = self.classes_[0]

        return self

    def _validate_y(self, y):
        # Default implementation
        return y


class LazyBaggingClassifier(BaseLazyBagging, ClassifierMixin):
    """A lazy bagging classifier.

    Everything is done lazily, models are built at prediction time and are not
    kept in memory. Since the models is thrown away, this allows to highly
    reduce the memory consumption and allows to build very large ensemble.

    Parameters
    ----------
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    Attributes
    ----------
    classes_ : array of shape = [n_classes]
        The classes labels.

    n_classes_ : int or list
        The number of classes.

    n_features_ : int,
        Number of features of the fitted input matrix

    n_outputs_ : int,
        Number of outputs of the fitted ouput matrix

    random_seed_ : int,
        Seed of the number generator


    References
    ----------
    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.

    """
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 random_state=None,
                 verbose=0):

        super(LazyBaggingClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            random_state=random_state,
            verbose=verbose)

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : array of shape = [n_samples, n_outputs]
            The predicted classes.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            n_samples = proba[0].shape[0]
            y = np.zeros((n_samples, self.n_outputs_))
            for k in range(self.n_outputs_):
                y[:, k] = self.classes_[k].take(np.argmax(proba[k], axis=1),
                                                axis=0)

            return y

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of a an input sample represents the proportion of estimators predicting
        each class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : array of shape = [n_samples, n_classes] or list of array
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        # Check data
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        n_samples, n_features = X.shape

        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        random_state = check_random_state(self.random_seed_)
        seeds = random_state.randint(MAX_INT, size=self.n_estimators)

        if self.n_outputs_ == 1:
            proba = np.zeros((n_samples, self.n_classes_))
        else:
            proba = [np.zeros((n_samples, self.n_classes_[k]))
                     for k in range(self.n_outputs_)]

        for out in _generator_fitted_estimators(self.n_estimators,
                                                self,
                                                self._X,
                                                self._y,
                                                self._sample_weight,
                                                seeds,
                                                self.verbose):
            estimator, _, features = out

            if self.n_outputs_ == 1:
                try:
                    proba_est = estimator.predict_proba(X[:, features])
                except AttributeError:
                    proba_est = np.zeros_like(proba)
                    y_pred = estimator.predict(X[:, features])
                    for c in range(self.n_classes_):
                        proba_est[:, c] = (y_pred == c).ravel()

                if self.n_classes_ == len(estimator.classes_):
                    proba += proba_est

                else:
                    for j, c in enumerate(estimator.classes_):
                        proba[:, c] += proba_est[:, j]

            else:
                try:
                    proba_est = estimator.predict_proba(X[:, features])
                except AttributeError:
                    y_pred = estimator.predict(X[:, features])
                    proba_est = []
                    for k, proba_k in enumerate(proba):
                        proba_est_k = np.zeros_like(proba_k)
                        for c in range(self.n_classes_[k]):
                            proba_est_k[:, c] = y_pred[:, k] == c

                        proba_est.append(proba_est_k)

                for k in range(self.n_outputs_):
                    if self.n_classes_[k] == len(estimator.classes_[k]):
                        proba[k] += proba_est[k]

                    else:
                        for j, c in enumerate(estimator.classes_[k]):
                            proba[k][:, c] += proba_est[k][:, j]

        # Divide by number of estimators
        if self.n_outputs_ == 1:
            proba /= self.n_estimators

        else:
            for k in range(self.n_outputs_):
                proba[k] /= self.n_estimators

        return proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the trees in the
        forest.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba

    def decision_function(self, X):
        """Average of the decision functions of the base classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        score : array, shape = [n_samples, k] or list of array
            The decision function of the input samples. The columns correspond
            to the classes in sorted order, as they appear in the attribute
            ``classes_``. Regression and binary classification are special
            cases with ``k == 1``, otherwise ``k==n_classes``.

        """
        raise NotImplementedError()


    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(LazyBaggingClassifier, self)._validate_estimator(
            default=DecisionTreeClassifier())

    def _validate_y(self, y):
        y = np.copy(y)
        self.classes_ = []
        self.n_classes_ = []

        for k in range(self.n_outputs_):
            classes_k, y[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])

        return y


class LazyBaggingRegressor(BaseLazyBagging, RegressorMixin):
    """A lazy bagging regressor.

    Everything is done lazily, models are built at prediction time and are not
    kept in memory. Since the models is thrown away, this allows to highly
    reduce the memory consumption and allows to build very large ensemble.

    Parameters
    ----------
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    Attributes
    ----------
    n_features_ : int,
        Number of features of the fitted input matrix

    n_outputs_ : int,
        Number of outputs of the fitted ouput matrix

    References
    ----------

    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.
    """
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 random_state=None,
                 verbose=0):
        super(LazyBaggingRegressor, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            random_state=random_state,
            verbose=verbose)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(LazyBaggingRegressor, self)._validate_estimator(
            default=DecisionTreeRegressor())

    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted values.
        """
        # Check data
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        n_samples, n_features = X.shape

        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        # Parallel loop
        random_state = check_random_state(self.random_seed_)
        seeds = random_state.randint(MAX_INT, size=self.n_estimators)

        if self.n_outputs_ == 1:
            y = np.zeros(n_samples)
        else:
            y = np.zeros((n_samples, self.n_outputs_))

        for out in _generator_fitted_estimators(self.n_estimators,
                                                self,
                                                self._X,
                                                self._y,
                                                self._sample_weight,
                                                seeds,
                                                self.verbose):
            estimator, _, features = out
            y += estimator.predict(X[:, features])


        # Normalize by the number of estimator
        y /= self.n_estimators

        return y
