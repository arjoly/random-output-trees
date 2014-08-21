=========
Reference
=========

This is the class and function reference of the package.


:mod:`randomized_output_forest.tree`: Provide tree-based estimator
------------------------------------------------------------------

This module provide tree-based estimators which worked transform output-space.

.. automodule:: randomized_output_forest.tree
   :no-members:
   :no-inherited-members:

.. currentmodule:: randomized_output_forest

.. autosummary::
   :toctree: generated/
   :template: class.rst

   tree.DecisionTreeClassifier
   tree.DecisionTreeRegressor


:mod:`randomized_output_forest.ensemble`: Provide ensemble based estimator
--------------------------------------------------------------------------

This module provide ensemble estimators which work transformed output-space.

.. automodule:: randomized_output_forest.ensemble
   :no-members:
   :no-inherited-members:

.. currentmodule:: randomized_output_forest

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ensemble.RandomForestClassifier
   ensemble.RandomForestRegressor
   ensemble.ExtraTreesClassifier
   ensemble.ExtraTreesRegressor


:mod:`randomized_output_forest.random_projection`: Dimensionality reduction methods based on random projection
--------------------------------------------------------------------------------------------------------------

.. automodule:: randomized_output_forest.random_projection
   :no-members:
   :no-inherited-members:

.. currentmodule:: randomized_output_forest

.. autosummary::
   :toctree: generated/
   :template: class.rst

   random_projection.RademacherRandomProjection
   random_projection.AchlioptasRandomProjection
   random_projection.SampledHadamardProjection
   random_projection.SampledIdentityProjection


:mod:`randomized_output_forest.transformer`: Provides general purpose meta-transformer
--------------------------------------------------------------------------------------

.. automodule:: randomized_output_forest.transformer
   :no-members:
   :no-inherited-members:

.. currentmodule:: randomized_output_forest

.. autosummary::
   :toctree: generated/
   :template: class.rst

   transformer.FixedStateTransformer
