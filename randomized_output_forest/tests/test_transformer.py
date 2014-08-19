from sklearn.base import BaseEstimator
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils import check_random_state

from sklearn.random_projection import GaussianRandomProjection
from randomized_output_forest.transformer import FixedStateTransformer

class IdentityProjection(BaseEstimator):

    def fit(self, X):
        return self

    def transform(self, X):
        return X


def test_fixed_state_transformer():

    random_state = check_random_state(0)
    X = random_state.rand(500, 100)

    # Check that setting the random_seed is equivalent to set the
    # random_state
    transf = GaussianRandomProjection(n_components=5, random_state=0)
    fixed_transf = FixedStateTransformer(
        GaussianRandomProjection(n_components=5), random_seed=0)
    assert_array_almost_equal(fixed_transf.fit_transform(X),
                              transf.fit_transform(X))

    # Check that set_params doesn't modify the results
    fixed_transf = FixedStateTransformer(
        GaussianRandomProjection(n_components=5, random_state=None))

    fixed_transf2 = FixedStateTransformer(
        GaussianRandomProjection(random_state=1, n_components=5))

    assert_array_almost_equal(fixed_transf.fit_transform(X),
                              fixed_transf2.fit_transform(X))

    # Check that it work when there is no random_state
    fixed_transf = FixedStateTransformer(IdentityProjection())
    assert_array_almost_equal(fixed_transf.fit_transform(X), X)
