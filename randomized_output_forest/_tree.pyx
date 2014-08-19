# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Arnaud Joly
#
# Licence: BSD 3 clause

from libc.stdlib cimport calloc, free, malloc, realloc
from libc.string cimport memcpy, memset
from libc.math cimport log as ln
from libc.math cimport floor



import numpy as np
cimport numpy as np
np.import_array()


# =============================================================================
# Types and constants
# =============================================================================

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE


# =============================================================================
# Scikit-learn import
# =============================================================================

# Criterion
from ._sklearn_tree import Criterion
from ._sklearn_tree cimport Criterion
from ._sklearn_tree import MSE

from ._sklearn_tree import Splitter
from ._sklearn_tree cimport Splitter
from ._sklearn_tree cimport SplitRecord

from ._sklearn_tree cimport SIZE_t
from ._sklearn_tree cimport DOUBLE_t
from ._sklearn_tree cimport DTYPE_t
from ._sklearn_tree cimport UINT32_t


cdef inline np.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size):
    """Encapsulate data into a 1D numpy array of intp's."""
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> size
    return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INTP, data)


# =============================================================================
# Custom Criterion
# =============================================================================

cdef class VarianceCriterion(Criterion):
    """Abstract criterion for regression.

    Computes variance of the target values left and right of the split point.
    Computation is linear in `n_samples` by using ::

        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples y_bar ** 2
    """
    cdef double* mean_left
    cdef double* mean_right
    cdef double* mean_total
    cdef double* sq_sum_left
    cdef double* sq_sum_right
    cdef double* sq_sum_total
    cdef double* var_left
    cdef double* var_right

    cdef double* sum_left
    cdef double* sum_right
    cdef double* sum_total

    cdef SIZE_t* n_classes
    cdef SIZE_t label_count_stride

    def __cinit__(self, SIZE_t n_outputs, np.ndarray[SIZE_t, ndim=1] n_classes):
        # Default values
        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        # Allocate accumulators
        self.mean_left = <double*> calloc(n_outputs, sizeof(double))
        self.mean_right = <double*> calloc(n_outputs, sizeof(double))
        self.mean_total = <double*> calloc(n_outputs, sizeof(double))
        self.sq_sum_left = <double*> calloc(n_outputs, sizeof(double))
        self.sq_sum_right = <double*> calloc(n_outputs, sizeof(double))
        self.sq_sum_total = <double*> calloc(n_outputs, sizeof(double))
        self.var_left = <double*> calloc(n_outputs, sizeof(double))
        self.var_right = <double*> calloc(n_outputs, sizeof(double))

        self.sum_left = <double*> calloc(n_outputs, sizeof(double))
        self.sum_right = <double*> calloc(n_outputs, sizeof(double))
        self.sum_total = <double*> calloc(n_outputs, sizeof(double))

        # Check for allocation errors
        if (self.mean_left == NULL or
                self.mean_right == NULL or
                self.mean_total == NULL or
                self.sq_sum_left == NULL or
                self.sq_sum_right == NULL or
                self.sq_sum_total == NULL or
                self.var_left == NULL or
                self.var_right == NULL or
                self.sum_left == NULL or
                self.sum_right == NULL or
                self.sum_total == NULL):
            raise MemoryError()


        # Count labels for each output
        self.n_classes = <SIZE_t*> malloc(n_outputs * sizeof(SIZE_t))
        if self.n_classes == NULL:
            raise MemoryError()

        cdef SIZE_t k = 0
        cdef SIZE_t label_count_stride = 0

        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

            if n_classes[k] > label_count_stride:
                label_count_stride = n_classes[k]

            if n_classes[k] > 2:
                raise ValueError("Implementation limited to binary "
                                 "classification")

        self.label_count_stride = label_count_stride


    def __dealloc__(self):
        """Destructor."""
        free(self.mean_left)
        free(self.mean_right)
        free(self.mean_total)
        free(self.sq_sum_left)
        free(self.sq_sum_right)
        free(self.sq_sum_total)
        free(self.var_left)
        free(self.var_right)

        free(self.sum_left)
        free(self.sum_right)
        free(self.sum_total)

        free(self.n_classes)

    def __reduce__(self):
        return (VarianceCriterion,
                (self.n_outputs,
                 sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)),
                 self.__getstate__())


    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef void init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                   double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                   SIZE_t end) nogil:
        """Initialize the criterion at node samples[start:end] and
        children samples[start:start] and samples[start:end]."""
        self.y = y
        self.y_stride = y_stride
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        cdef double weighted_n_node_samples = 0.

        # Initialize accumulators
        cdef SIZE_t n_outputs = self.n_outputs
        cdef double* mean_left = self.mean_left
        cdef double* mean_right = self.mean_right
        cdef double* mean_total = self.mean_total
        cdef double* sq_sum_left = self.sq_sum_left
        cdef double* sq_sum_right = self.sq_sum_right
        cdef double* sq_sum_total = self.sq_sum_total
        cdef double* var_left = self.var_left
        cdef double* var_right = self.var_right
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef SIZE_t i = 0
        cdef SIZE_t p = 0
        cdef SIZE_t k = 0
        cdef DOUBLE_t y_ik = 0.0
        cdef DOUBLE_t w_y_ik = 0.0
        cdef DOUBLE_t w = 1.0

        cdef SIZE_t n_bytes = n_outputs * sizeof(double)
        memset(mean_left, 0, n_bytes)
        memset(mean_right, 0, n_bytes)
        memset(mean_total, 0, n_bytes)
        memset(sq_sum_left, 0, n_bytes)
        memset(sq_sum_right, 0, n_bytes)
        memset(sq_sum_total, 0, n_bytes)
        memset(var_left, 0, n_bytes)
        memset(var_right, 0, n_bytes)
        memset(sum_left, 0, n_bytes)
        memset(sum_right, 0, n_bytes)
        memset(sum_total, 0, n_bytes)

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(n_outputs):
                y_ik = y[i * y_stride + k]
                w_y_ik = w * y_ik
                sum_total[k] += w_y_ik
                sq_sum_total[k] += w_y_ik * y_ik

            weighted_n_node_samples += w

        self.weighted_n_node_samples = weighted_n_node_samples

        for k in range(n_outputs):
            mean_total[k] = sum_total[k] / weighted_n_node_samples

        # Reset to pos=start
        self.reset()

    cdef void reset(self) nogil:
        """Reset the criterion at pos=start."""
        self.pos = self.start

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        cdef double weighted_n_right = self.weighted_n_right


        cdef SIZE_t n_outputs = self.n_outputs
        cdef double* mean_left = self.mean_left
        cdef double* mean_right = self.mean_right
        cdef double* mean_total = self.mean_total
        cdef double* sq_sum_left = self.sq_sum_left
        cdef double* sq_sum_right = self.sq_sum_right
        cdef double* sq_sum_total = self.sq_sum_total
        cdef double* var_left = self.var_left
        cdef double* var_right = self.var_right
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef SIZE_t k = 0

        for k in range(n_outputs):
            mean_right[k] = mean_total[k]
            mean_left[k] = 0.0
            sq_sum_right[k] = sq_sum_total[k]
            sq_sum_left[k] = 0.0
            var_right[k] = (sq_sum_right[k] / weighted_n_right -
                            mean_right[k] * mean_right[k])
            var_left[k] = 0.0
            sum_right[k] = sum_total[k]
            sum_left[k] = 0.0


    cdef void update(self, SIZE_t new_pos) nogil:
        """Update the collected statistics by moving samples[pos:new_pos] from
           the right child to the left child."""
        cdef DOUBLE_t* y = self.y
        cdef SIZE_t y_stride = self.y_stride
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos

        cdef SIZE_t n_outputs = self.n_outputs
        cdef double* mean_left = self.mean_left
        cdef double* mean_right = self.mean_right
        cdef double* sq_sum_left = self.sq_sum_left
        cdef double* sq_sum_right = self.sq_sum_right
        cdef double* var_left = self.var_left
        cdef double* var_right = self.var_right
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef double weighted_n_left = self.weighted_n_left
        cdef double weighted_n_right = self.weighted_n_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t y_ik, w_y_ik

        # Note: We assume start <= pos < new_pos <= end
        for p in range(pos, new_pos):
            i = samples[p]

            if sample_weight != NULL:
                w  = sample_weight[i]

            for k in range(n_outputs):
                y_ik = y[i * y_stride + k]
                w_y_ik = w * y_ik

                sum_left[k] += w_y_ik
                sum_right[k] -= w_y_ik

                sq_sum_left[k] += w_y_ik * y_ik
                sq_sum_right[k] -= w_y_ik * y_ik

            weighted_n_left += w
            weighted_n_right -= w

        for k in range(n_outputs):
            mean_left[k] = sum_left[k] / weighted_n_left
            mean_right[k] = sum_right[k] / weighted_n_right
            var_left[k] = (sq_sum_left[k] / weighted_n_left -
                           mean_left[k] * mean_left[k])
            var_right[k] = (sq_sum_right[k] / weighted_n_right -
                            mean_right[k] * mean_right[k])

        self.weighted_n_left = weighted_n_left
        self.weighted_n_right = weighted_n_right

        self.pos = new_pos

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""
        cdef SIZE_t n_outputs = self.n_outputs
        cdef double* sq_sum_total = self.sq_sum_total
        cdef double* mean_total = self.mean_total
        cdef double weighted_n_node_samples = self.weighted_n_node_samples
        cdef double total = 0.0
        cdef SIZE_t k

        for k in range(n_outputs):
            total += (sq_sum_total[k] / weighted_n_node_samples -
                      mean_total[k] * mean_total[k])

        return total / n_outputs

    cdef void children_impurity(self, double* impurity_left, double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""
        cdef SIZE_t n_outputs = self.n_outputs
        cdef double* var_left = self.var_left
        cdef double* var_right = self.var_right
        cdef double total_left = 0.0
        cdef double total_right = 0.0
        cdef SIZE_t k

        for k in range(n_outputs):
            total_left += var_left[k]
            total_right += var_right[k]

        impurity_left[0] = total_left / n_outputs
        impurity_right[0] = total_right / n_outputs

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride

        cdef DOUBLE_t* y = self.y
        cdef SIZE_t y_stride = self.y_stride
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t c
        cdef DOUBLE_t w = 1.

        cdef SIZE_t i = 0
        cdef SIZE_t p = 0
        cdef SIZE_t k = 0
        cdef SIZE_t offset = 0

        for k in range(n_outputs):
            memset(dest + offset, 0, n_classes[k] * sizeof(double))
            offset += label_count_stride

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(n_outputs):
                c = <SIZE_t> y[i * y_stride + k]
                dest[k * label_count_stride + c] += w



# =============================================================================
# Custom splitter
# =============================================================================

cdef class SplitterTransformer(Splitter):
    """Base splitter for working on a transformed space"""

    cdef Splitter splitter                 # Splitter used for the data

    cdef np.ndarray y_transformed
    cdef DOUBLE_t* y_transformed_data      # Storage of transformed output
    cdef SIZE_t y_transformed_stride  # Stride of transformed output


    def __getstate__(self):
        return {"splitter": self.splitter,
                "y_transformed": self.y_transformed}

    def __setstate__(self, d):
        self.set_output_space(d["splitter"], d["y_transformed"])

    def __reduce__(self):
        return (SplitterTransformer, (self.criterion,
                                      self.max_features,
                                      self.min_samples_leaf,
                                      self.min_weight_leaf,
                                      self.random_state), self.__getstate__())


    def set_output_space(self,
                         Splitter splitter,
                         np.ndarray[DOUBLE_t, ndim=2,  mode="c"] y):

        # Set transformed output space
        self.y_transformed = y
        self.y_transformed_data = <DOUBLE_t*> y.data
        self.y_transformed_stride = (<SIZE_t> y.strides[0] /
                                     <SIZE_t> y.itemsize)

        # Set sub-splitter and its criterion
        self.splitter = splitter


    cdef void init(self, np.ndarray[DTYPE_t, ndim=2] X,
                         np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                         DOUBLE_t* sample_weight) except *:
        """Initialize the splitter."""

        if not self.splitter:
            raise ValueError('Unspecify base splitter')

        if self.y_transformed_data == NULL:
            raise ValueError("Unspectify subspace use set_output_space")

        # Initialize the splitter
        self.splitter.criterion = MSE(self.y_transformed.shape[1])
        self.splitter.init(X, self.y_transformed, sample_weight)
        self.n_samples = self.splitter.n_samples


        # State of the splitter
        self.y = <DOUBLE_t*> y.data
        self.y_stride = (<SIZE_t> y.strides[0] / <SIZE_t> y.itemsize)


    cdef void node_reset(self, SIZE_t start, SIZE_t end,
                         double* weighted_n_node_samples) nogil:
        """Reset splitter on node samples[start:end]."""
        # Reset the base splitter
        self.start = start
        self.end = end
        self.splitter.node_reset(start, end, weighted_n_node_samples)

    cdef void node_split(self, double impurity,
                         SplitRecord* split,
                         SIZE_t* n_constant_features) nogil:
        """Find a split on node samples[start:end]."""
        self.splitter.node_split(impurity, split, n_constant_features)

    cdef void node_value(self, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest."""
        self.criterion.init(self.y,
                            self.y_stride,
                            self.splitter.sample_weight,
                            self.splitter.weighted_n_samples,
                            self.splitter.samples,
                            self.splitter.start,
                            self.splitter.end)
        self.criterion.node_value(dest)

    cdef double node_impurity(self) nogil:
        """Impurity at the node"""
        return self.splitter.node_impurity()
