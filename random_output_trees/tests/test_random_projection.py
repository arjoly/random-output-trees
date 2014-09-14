
import numpy as np
from scipy.sparse import issparse, coo_matrix, csr_matrix
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_warns

from random_output_trees.random_projection import RademacherRandomProjection
from random_output_trees.random_projection import AchlioptasRandomProjection
from random_output_trees.random_projection import SampledHadamardProjection
from random_output_trees.random_projection import SampledIdentityProjection

from random_output_trees.random_projection import subsampled_hadamard_matrix
from random_output_trees.random_projection import subsampled_identity_matrix


RANDOM_PROJECTION = {
    "RademacherRandomProjection": RademacherRandomProjection,
    "AchlioptasRandomProjection": AchlioptasRandomProjection,
    "SampledHadamardProjection": SampledHadamardProjection,
    "SampledIdentityProjection": SampledIdentityProjection,
}

all_random_matrix = {
    "subsample_hadamard_matrix": subsampled_hadamard_matrix,
    "random_subsample_normalized": subsampled_identity_matrix,
}

def make_sparse_random_data(n_samples, n_features, n_nonzeros):
    rng = np.random.RandomState(0)
    data_coo = coo_matrix(
        (rng.randn(n_nonzeros),
         (rng.randint(n_samples, size=n_nonzeros),
          rng.randint(n_features, size=n_nonzeros))),
        shape=(n_samples, n_features))
    return data_coo.toarray(), data_coo.tocsr()

n_samples, n_features = (10, 1000)
n_nonzeros = int(n_samples * n_features / 100.)
data, data_csr = make_sparse_random_data(n_samples, n_features, n_nonzeros)

def densify(matrix):
    if not issparse(matrix):
        return matrix
    else:
        return matrix.toarray()

def check_random_projection(name):
    RandomProjection = RANDOM_PROJECTION[name]

    # Invalid input
    assert_raises(ValueError, RandomProjection(n_components='auto').fit,
                  [0, 1, 2])
    assert_raises(ValueError, RandomProjection(n_components=-10).fit, data)

    # Try to transform before fit
    assert_raises(ValueError, RandomProjection(n_components='auto').transform,
                  data)


def test_too_many_samples_to_find_a_safe_embedding():
    data, _ = make_sparse_random_data(1000, 100, 1000)

    for name, RandomProjection in RANDOM_PROJECTION.items():
        rp = RandomProjection(n_components='auto', eps=0.1)
        expected_msg = (
            'eps=0.100000 and n_samples=1000 lead to a target dimension'
            ' of 5920 which is larger than the original space with'
            ' n_features=100')
        assert_raise_message(ValueError, expected_msg, rp.fit, data)



def test_correct_RandomProjection_dimensions_embedding():
    for name, RandomProjection in RANDOM_PROJECTION.items():
        rp = RandomProjection(n_components='auto',
                              random_state=0,
                              eps=0.5).fit(data)

        # the number of components is adjusted from the shape of the training
        # set
        assert_equal(rp.n_components, 'auto')
        assert_equal(rp.n_components_, 110)

        assert_equal(rp.components_.shape, (110, n_features))

        projected_1 = rp.transform(data)
        assert_equal(projected_1.shape, (n_samples, 110))

        # once the RP is 'fitted' the projection is always the same
        projected_2 = rp.transform(data)
        assert_array_equal(projected_1, projected_2)

        # fit transform with same random seed will lead to the same results
        rp2 = RandomProjection(random_state=0, eps=0.5)
        projected_3 = rp2.fit_transform(data)
        assert_array_equal(projected_1, projected_3)

        # Try to transform with an input X of size different from fitted.
        assert_raises(ValueError, rp.transform, data[:, 1:5])


def test_warning_n_components_greater_than_n_features():
    n_features = 20
    data, _ = make_sparse_random_data(5, n_features, int(n_features / 4))

    for name, RandomProjection in RANDOM_PROJECTION.items():
        assert_warns(UserWarning,
                     RandomProjection(n_components=n_features + 1).fit, data)


def test_works_with_sparse_data():
    n_features = 20
    data, _ = make_sparse_random_data(5, n_features, int(n_features / 4))

    for name, RandomProjection in RANDOM_PROJECTION.items():
        rp_dense = RandomProjection(n_components=3,
                                    random_state=1).fit(data)
        rp_sparse = RandomProjection(n_components=3,
                                     random_state=1).fit(csr_matrix(data))
        assert_array_almost_equal(densify(rp_dense.components_),
                                  densify(rp_sparse.components_))


###############################################################################
# tests random matrix generation
###############################################################################
def check_input_size_random_matrix(random_matrix):
    assert_raises(ValueError, random_matrix, 0, 0)
    assert_raises(ValueError, random_matrix, -1, 1)
    assert_raises(ValueError, random_matrix, 1, -1)
    assert_raises(ValueError, random_matrix, 1, 0)
    assert_raises(ValueError, random_matrix, -1, 0)


def check_size_generated(random_matrix):
    assert_equal(random_matrix(1, 5).shape, (1, 5))
    assert_equal(random_matrix(5, 1).shape, (5, 1))
    assert_equal(random_matrix(5, 5).shape, (5, 5))
    assert_equal(random_matrix(1, 1).shape, (1, 1))


def check_zero_mean_and_unit_norm(random_matrix):
    # All random matrix should produce a transformation matrix
    # with zero mean and unit norm for each columns

    A = densify(random_matrix(1000, 1, random_state=0)).ravel()
    assert_array_almost_equal(0, np.mean(A), 3)
    assert_array_almost_equal(1.0, np.linalg.norm(A),  1)


def check_approximate_isometry(random_matrix):
    A =  densify(random_matrix(50, 10, 0))
    assert_almost_equal(np.mean(np.diag(np.dot(A.T, A))), 1.)

def test_basic_property_of_random_matrix():
    """Check basic properties of random matrix generation"""
    for name, random_matrix in all_random_matrix.items():
        print(name)

        check_input_size_random_matrix(random_matrix)
        check_size_generated(random_matrix)
        if name != "random_subsample_normalized":
            check_zero_mean_and_unit_norm(random_matrix)
        check_approximate_isometry(random_matrix)

