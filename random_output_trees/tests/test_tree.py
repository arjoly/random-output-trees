"""
Testing for the tree module.
"""
from functools import partial

from sklearn import datasets
from sklearn.cross_validation import train_test_split

from sklearn.decomposition import PCA


from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_equal

from random_output_trees.tree import DecisionTreeClassifier
from random_output_trees.tree import DecisionTreeRegressor

from sklearn.random_projection import GaussianRandomProjection
from sklearn.base import BaseEstimator, TransformerMixin

class IdentityProjection(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self

    def transform(self, X):
        return X


CLF_TREES = {
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "Presort-DecisionTreeClassifier": partial(DecisionTreeClassifier,
                                              splitter="presort-best"),
    "ExtraTreeClassifier": partial(DecisionTreeClassifier,
                                              splitter="random"),
}

REG_TREES = {
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "Presort-DecisionTreeRegressor": partial(DecisionTreeRegressor,
                                             splitter="presort-best"),
    "ExtraTreeRegressor": partial(DecisionTreeRegressor,
                                  splitter="random"),
}

ALL_TREES = dict()
ALL_TREES.update(CLF_TREES)
ALL_TREES.update(REG_TREES)


def test_output_transformer():
    X, y = datasets.make_multilabel_classification(return_indicator=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    transformer = GaussianRandomProjection(n_components=10)
    for name, TreeEstimator in ALL_TREES.items():
        est = TreeEstimator(random_state=0, output_transformer=transformer)
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        assert_equal(y_pred.shape, y_test.shape)


def test_identity_output_transformer():

    X, y = datasets.make_multilabel_classification(return_indicator=True,
        random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    transformer = IdentityProjection()

    for name, TreeEstimator in ALL_TREES.items():
        est = TreeEstimator(random_state=0, max_features=None)
        est.fit(X_train, y_train)
        y_pred_origin = est.predict(X_test)


        est_transf = TreeEstimator(random_state=0, max_features=None,
                                   output_transformer=transformer)
        est_transf.fit(X_train, y_train)
        y_pred_transformed = est_transf.predict(X_test)
        assert_almost_equal(y_pred_origin, y_pred_transformed, decimal=5,
                            err_msg="failed with {0}".format(name))


def test_pca_output_transformer():
    X, y = datasets.make_multilabel_classification(return_indicator=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    transformer = PCA(n_components=1)

    for name, TreeEstimator in ALL_TREES.items():
        est_transf = TreeEstimator(random_state=0,
                                   max_features=None,
                                   output_transformer=transformer)
        est_transf.fit(X_train, y_train)
        y_pred_transformed = est_transf.predict(X_test)
        assert_equal(y_pred_transformed.shape, y_test.shape,
                     msg="failed with {0}".format(name))


def test_importances_variance_equal_mse():
    """Check that gini is equivalent to mse for binary output variable"""

    from sklearn.tree._tree import TREE_LEAF

    X, y = datasets.make_classification(n_samples=2000,
                                        n_features=10,
                                        n_informative=3,
                                        n_redundant=0,
                                        n_repeated=0,
                                        shuffle=False,
                                        random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


    var = DecisionTreeClassifier(criterion="variance",
                                 random_state=0).fit(X_train, y_train)
    gini = DecisionTreeClassifier(criterion="gini",
                                  random_state=0).fit(X_train, y_train)
    reg = DecisionTreeRegressor(criterion="mse",
                                random_state=0).fit(X_train, y_train)

    gini_leaves = gini.tree_.children_left == TREE_LEAF
    var_leaves = var.tree_.children_left == TREE_LEAF

    assert_array_equal(var.tree_.feature, reg.tree_.feature)
    assert_almost_equal(var.feature_importances_, reg.feature_importances_)
    assert_array_equal(var.tree_.children_left, reg.tree_.children_left)
    assert_array_equal(var.tree_.children_right, reg.tree_.children_right)
    assert_array_equal(var.tree_.n_node_samples, reg.tree_.n_node_samples)

    assert_array_equal(var.tree_.feature, gini.tree_.feature)
    assert_almost_equal(var.feature_importances_, gini.feature_importances_)
    assert_array_equal(var.tree_.children_left, gini.tree_.children_left)
    assert_array_equal(var.tree_.children_right, gini.tree_.children_right)
    assert_array_equal(var.tree_.n_node_samples, gini.tree_.n_node_samples)
    assert_almost_equal(var.tree_.value[var_leaves], gini.tree_.value[gini_leaves])


    clf = DecisionTreeClassifier(criterion="gini", random_state=0,
                                  output_transformer=IdentityProjection(),
                                 ).fit(X_train, y_train)

    clf_leaves = clf.tree_.children_left == TREE_LEAF
    assert_array_equal(clf.tree_.feature, reg.tree_.feature)
    assert_almost_equal(clf.feature_importances_, reg.feature_importances_)
    assert_array_equal(clf.tree_.children_left, reg.tree_.children_left)
    assert_array_equal(clf.tree_.children_right, reg.tree_.children_right)
    assert_array_equal(clf.tree_.n_node_samples, reg.tree_.n_node_samples)
    assert_array_equal(clf.tree_.n_node_samples, reg.tree_.n_node_samples)

    assert_array_equal(clf.tree_.feature, gini.tree_.feature)
    assert_almost_equal(clf.feature_importances_, gini.feature_importances_)
    assert_array_equal(clf.tree_.children_left, gini.tree_.children_left)
    assert_array_equal(clf.tree_.children_right, gini.tree_.children_right)
    assert_array_equal(clf.tree_.n_node_samples, gini.tree_.n_node_samples)
    assert_almost_equal(clf.tree_.value[clf_leaves], gini.tree_.value[gini_leaves])
