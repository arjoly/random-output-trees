"""
Testing for the forest module (sklearn.ensemble.forest).
"""

# Most tests comes from scikit-learn and ensure that everything is working
# as expected

# Authors: Gilles Louppe,
#          Brian Holt,
#          Andreas Mueller,
#          Arnaud Joly
# License: BSD 3 clause



from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_equal

from sklearn import datasets
from sklearn.utils.validation import check_random_state

from sklearn.cross_validation import train_test_split
from sklearn.random_projection import GaussianRandomProjection
from sklearn.base import BaseEstimator, TransformerMixin

from randomized_output_forest.transformer import FixedStateTransformer

from randomized_output_forest.ensemble import ExtraTreesClassifier
from randomized_output_forest.ensemble import ExtraTreesRegressor
from randomized_output_forest.ensemble import RandomForestClassifier
from randomized_output_forest.ensemble import RandomForestRegressor


# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = check_random_state(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# also load the boston dataset
# and randomly permute it
boston = datasets.load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]

FOREST_CLASSIFIERS = {
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "RandomForestClassifier": RandomForestClassifier,
}

FOREST_REGRESSORS = {
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "RandomForestRegressor": RandomForestRegressor,
}

FOREST_TRANSFORMERS = {}

FOREST_ESTIMATORS = dict()
FOREST_ESTIMATORS.update(FOREST_CLASSIFIERS)
FOREST_ESTIMATORS.update(FOREST_REGRESSORS)
FOREST_ESTIMATORS.update(FOREST_TRANSFORMERS)


class IdentityProjections(BaseEstimator, TransformerMixin):
    """ Project the input data on the identity matrix (noop operation)"""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(selft, X):
        return X


def test_output_transformer():
    X, y = datasets.make_multilabel_classification(return_indicator=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Check that random_state are different
    transformer = GaussianRandomProjection(n_components=5, random_state=None)
    for name, ForestEstimator in FOREST_ESTIMATORS.items():
        est = ForestEstimator(random_state=5, output_transformer=transformer)
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        assert_equal(y_pred.shape, y_test.shape)

        random_state = [sub.output_transformer_.random_state
                        for sub in est.estimators_]

        assert_equal(len(set(random_state)), est.n_estimators)


    # Check that random_state are equals
    transformer = FixedStateTransformer(GaussianRandomProjection(
        n_components=5), random_seed=0)
    for name, ForestEstimator in FOREST_ESTIMATORS.items():
        est = ForestEstimator(random_state=5, output_transformer=transformer)
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        assert_equal(y_pred.shape, y_test.shape)


        random_state = [sub.output_transformer_.random_state
                        for sub in est.estimators_]

        assert_equal(len(set(random_state)), 1)
        assert_equal(random_state[0], 0)


def test_identity_output_transformer():
    X, y = datasets.make_multilabel_classification(return_indicator=True,
                                                   random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    for name, ForestEstimator in FOREST_ESTIMATORS.items():
        est = ForestEstimator(random_state=0, max_features=None, max_depth=4)
        est.fit(X_train, y_train)
        y_pred_origin = est.predict(X_test)


        est_transf = est.set_params(output_transformer=IdentityProjections())
        est_transf.fit(X_train, y_train)
        y_pred_transformed = est_transf.predict(X_test)
        assert_almost_equal(y_pred_origin, y_pred_transformed)


if __name__ == "__main__":
    import nose
    nose.runmodule()
