import numpy as np

from scipy.linalg import hadamard as sp_hadamard
from scipy import sparse

from sklearn.random_projection import BaseRandomProjection
from sklearn.random_projection import SparseRandomProjection

from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import check_random_state

__all__ = [
    "RademacherRandomProjection",
    "AchlioptasRandomProjection",
    "SampledHadamardProjection",
    "SampledIdentityProjection",
]


class RademacherRandomProjection(SparseRandomProjection):
    """Rademacher random projection

    The components of the random matrix
    are drawn from:

      - -sqrt(s) / sqrt(n_components)   with probability 1 / 2
      - +sqrt(s) / sqrt(n_components)   with probability 1 / 2

    Parameters
    ----------
    n_components : int or 'auto', optional (default = 'auto')
        Dimensionality of the target projection space.

        n_components can be automatically adjusted according to the
        number of samples in the dataset and the bound given by the
        Johnson-Lindenstrauss lemma. In that case the quality of the
        embedding is controlled by the ``eps`` parameter.

        It should be noted that Johnson-Lindenstrauss lemma can yield
        very conservative estimated of the required number of components
        as it makes no assumption on the structure of the dataset.

    eps : strictly positive float, optional (default=0.1)
        Parameter to control the quality of the embedding according to
        the Johnson-Lindenstrauss lemma when n_components is set to
        'auto'.

        Smaller values lead to better embedding and higher number of
        dimensions (n_components) in the target projection space.

    random_state : integer, RandomState instance or None (default=None)
        Control the pseudo random number generator used to generate the
        matrix at fit time.

    Attributes
    ----------
    ``n_component_`` : int
        Concrete number of components computed when n_components="auto".

    ``components_`` : numpy array of shape [n_components, n_features]
        Random matrix used for the projection.

    """
    def __init__(self, n_components="auto", eps=0.1, random_state=None):
        super(RademacherRandomProjection, self).__init__(
            n_components=n_components,
            eps=eps,
            density=1,
            dense_output=True,
            random_state=random_state)


class AchlioptasRandomProjection(SparseRandomProjection):
    """Sparse random projection using Achlioptas random matrix

    If we note `s = 1 / density = 1 / 3 ` the components of the random matrix
    are drawn from:

      - -sqrt(s) / sqrt(n_components)   with probability 1 / 2s
      -  0                              with probability 1 - 1 / s
      - +sqrt(s) / sqrt(n_components)   with probability 1 / 2s

    Parameters
    ----------
    n_components : int or 'auto', optional (default = 'auto')
        Dimensionality of the target projection space.

        n_components can be automatically adjusted according to the
        number of samples in the dataset and the bound given by the
        Johnson-Lindenstrauss lemma. In that case the quality of the
        embedding is controlled by the ``eps`` parameter.

        It should be noted that Johnson-Lindenstrauss lemma can yield
        very conservative estimated of the required number of components
        as it makes no assumption on the structure of the dataset.

    eps : strictly positive float, optional (default=0.1)
        Parameter to control the quality of the embedding according to
        the Johnson-Lindenstrauss lemma when n_components is set to
        'auto'.

        Smaller values lead to better embedding and higher number of
        dimensions (n_components) in the target projection space.

    dense_output : boolean, optional (default=False)
        If True, ensure that the output of the random projection is a
        dense numpy array even if the input and random projection matrix
        are both sparse. In practice, if the number of components is
        small the number of zero components in the projected data will
        be very small and it will be more CPU and memory efficient to
        use a dense representation.

        If False, the projected data uses a sparse representation if
        the input is sparse.

    random_state : integer, RandomState instance or None (default=None)
        Control the pseudo random number generator used to generate the
        matrix at fit time.

    Attributes
    ----------
    ``n_component_`` : int
        Concrete number of components computed when n_components="auto".

    ``components_`` : numpy array of shape [n_components, n_features]
        Random matrix used for the projection.

    """
    def __init__(self, n_components="auto", eps=0.1, random_state=None,
                 dense_output=False):
        super(AchlioptasRandomProjection, self).__init__(
            n_components=n_components,
            eps=eps,
            density=1. / 3,
            dense_output=dense_output,
            random_state=random_state)


def subsampled_hadamard_matrix(n_components, n_features, random_state=None):
    """Sub-sampled hadamard matrix to have shape n_components and n_features

    A hadamard matrix of shape at (least n_components, n_features) is
    subsampled without replacement.

    Parameters
    ----------
    n_components : int,
        Dimensionality of the target projection space.

    n_features : int,
        Dimensionality of the original source space.

    random_state : int, RandomState instance or None (default=None)
        Control the pseudo random number generator used to generate the
        matrix at fit time.

    Returns
    -------
    components : numpy array of shape [n_components, n_features]
        The generated random matrix.

    """
    if n_components <= 0:
        raise ValueError("n_components must be strictly positive, got %d" %
                         n_components)
    if n_features <= 0:
        raise ValueError("n_features must be strictly positive, got %d" %
                         n_components)

    random_state = check_random_state(random_state)
    n_hadmard_size = max(2 ** np.ceil(np.log2(x))
                         for x in (n_components, n_features))

    row = sample_without_replacement(n_hadmard_size, n_components,
                                     random_state=random_state)
    col = sample_without_replacement(n_hadmard_size, n_features,
                                     random_state=random_state)
    hadamard_matrix = sp_hadamard(n_hadmard_size, dtype=np.float)[row][:, col]
    hadamard_matrix *= 1 / np.sqrt(n_components)
    return hadamard_matrix


class SampledHadamardProjection(BaseRandomProjection):
    """Subsample Hadamard random projection

    The components of the random matrix are obtnained by subsampling the
    row and column of a sufficiently big Hadamard matrix.

    Parameters
    ----------
    n_components : int or 'auto', optional (default = 'auto')
        Dimensionality of the target projection space.

        n_components can be automatically adjusted according to the
        number of samples in the dataset and the bound given by the
        Johnson-Lindenstrauss lemma. In that case the quality of the
        embedding is controlled by the ``eps`` parameter.

        It should be noted that Johnson-Lindenstrauss lemma can yield
        very conservative estimated of the required number of components
        as it makes no assumption on the structure of the dataset.

    eps : strictly positive float, optional (default=0.1)
        Parameter to control the quality of the embedding according to
        the Johnson-Lindenstrauss lemma when n_components is set to
        'auto'.

        Smaller values lead to better embedding and higher number of
        dimensions (n_components) in the target projection space.

    random_state : integer, RandomState instance or None (default=None)
        Control the pseudo random number generator used to generate the
        matrix at fit time.

    Attributes
    ----------
    ``n_component_`` : int
        Concrete number of components computed when n_components="auto".

    ``components_`` : numpy array of shape [n_components, n_features]
        Random matrix used for the projection.

    """
    def __init__(self, n_components="auto", eps=0.1, random_state=None):
        super(SampledHadamardProjection, self).__init__(
            n_components=n_components,
            eps=eps,
            random_state=random_state)

    def _make_random_matrix(self, n_components, n_features):
        return subsampled_hadamard_matrix(n_components, n_features,
                                          random_state=self.random_state)


def subsampled_identity_matrix(n_components, n_features, random_state=None):
    """Sub-sampled identity matrix to have shape n_components and n_features

    Parameters
    ----------
    n_components : int,
        Dimensionality of the target projection space.

    n_features : int,
        Dimensionality of the original source space.

    random_state : int, RandomState instance or None (default=None)
        Control the pseudo random number generator used to generate the
        matrix at fit time.

    Returns
    -------
    components : numpy array of shape [n_components, n_features]
        The generated random matrix.

    """

    if n_components <= 0:
        raise ValueError("n_components must be strictly positive, got %d" %
                         n_components)
    if n_features <= 0:
        raise ValueError("n_features must be strictly positive, got %d" %
                         n_components)

    rng = check_random_state(random_state)

    components = sparse.dia_matrix((np.ones(n_features), [0]),
                                   shape=(n_features, n_features)).tocsr()
    components = components[rng.randint(n_features, size=(n_components,))]
    return components * np.sqrt(1.0 * n_features / n_components)


class SampledIdentityProjection(BaseRandomProjection):
    """Subsample identity matrix projection

    The components of the random matrix are obtnained by subsampling the
    row and column of the identity matrix.

    Parameters
    ----------
    n_components : int or 'auto', optional (default = 'auto')
        Dimensionality of the target projection space.

        n_components can be automatically adjusted according to the
        number of samples in the dataset and the bound given by the
        Johnson-Lindenstrauss lemma. In that case the quality of the
        embedding is controlled by the ``eps`` parameter.

        It should be noted that Johnson-Lindenstrauss lemma can yield
        very conservative estimated of the required number of components
        as it makes no assumption on the structure of the dataset.

    eps : strictly positive float, optional (default=0.1)
        Parameter to control the quality of the embedding according to
        the Johnson-Lindenstrauss lemma when n_components is set to
        'auto'.

        Smaller values lead to better embedding and higher number of
        dimensions (n_components) in the target projection space.

        Note that the JL-lemma is not appropriate for the projection of a
        sample identity projection.

    random_state : integer, RandomState instance or None (default=None)
        Control the pseudo random number generator used to generate the
        matrix at fit time.

    Attributes
    ----------
    ``n_component_`` : int
        Concrete number of components computed when n_components="auto".

    ``components_`` : numpy array of shape [n_components, n_features]
        Random matrix used for the projection.

    """
    def __init__(self, n_components="auto", eps=0.1, random_state=None,
                 dense_output=False):
        super(SampledIdentityProjection, self).__init__(
            n_components=n_components,
            eps=eps,
            dense_output=dense_output,
            random_state=random_state)

    def _make_random_matrix(self, n_components, n_features):
        return subsampled_identity_matrix(n_components, n_features,
                                          self.random_state)
