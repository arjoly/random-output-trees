'''
This module provides ensemble estimators which work transformed output-space.

'''

from .forest import RandomForestClassifier
from .forest import RandomForestRegressor
from .forest import ExtraTreesClassifier
from .forest import ExtraTreesRegressor
from .lazy_bagging import LazyBaggingClassifier
from .lazy_bagging import LazyBaggingRegressor

__all__ = [
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
]
