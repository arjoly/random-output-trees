==========
References
==========

This is the class and function reference of the package.


:mod:`random_output_trees.ensemble`: Ensemble
---------------------------------------------

This module provides ensemble estimators which work transformed output-space.

.. automodule:: random_output_trees.ensemble
   :no-members:
   :no-inherited-members:

.. currentmodule:: random_output_trees

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ensemble.RandomForestClassifier
   ensemble.RandomForestRegressor
   ensemble.ExtraTreesClassifier
   ensemble.ExtraTreesRegressor


:mod:`random_output_trees.datasets`: Datasets
---------------------------------------------

This module provides dataset loaders and fetchers.


.. automodule:: random_output_trees.ensemble
   :no-members:
   :no-inherited-members:

.. currentmodule:: random_output_trees

.. autosummary::
   :toctree: generated/
   :template: function.rst

   datasets.fetch_drug_interaction
   datasets.fetch_protein_interaction


:mod:`random_output_trees.random_projection`: Random projection
---------------------------------------------------------------

This module provides dimensionality reduction methods based on random
projection.

.. automodule:: random_output_trees.random_projection
   :no-members:
   :no-inherited-members:

.. currentmodule:: random_output_trees

.. autosummary::
   :toctree: generated/
   :template: class.rst

   random_projection.RademacherRandomProjection
   random_projection.AchlioptasRandomProjection
   random_projection.SampledHadamardProjection
   random_projection.SampledIdentityProjection


:mod:`random_output_trees.transformer`: Transformer
---------------------------------------------------

This module provides general purpose meta-transformer.

.. automodule:: random_output_trees.transformer
   :no-members:
   :no-inherited-members:

.. currentmodule:: random_output_trees

.. autosummary::
   :toctree: generated/
   :template: class.rst

   transformer.FixedStateTransformer


:mod:`random_output_trees.tree`: Tree
-------------------------------------

This module provide tree-based estimators which worked transform output-space.

.. automodule:: random_output_trees.tree
   :no-members:
   :no-inherited-members:

.. currentmodule:: random_output_trees

.. autosummary::
   :toctree: generated/
   :template: class.rst

   tree.DecisionTreeClassifier
   tree.DecisionTreeRegressor
