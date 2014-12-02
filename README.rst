Random output trees
===================

.. image:: https://secure.travis-ci.org/arjoly/andom-output-trees.png?branch=master
   :target: https://secure.travis-ci.org/arjoly/random-output-trees
   :alt: Build status

.. image:: https://coveralls.io/repos/arjoly/andom-output-trees/badge.png?branch=master
   :target: https://coveralls.io/r/arjoly/random-output-trees
   :alt: Coverage status

.. image:: https://landscape.io/github/arjoly/random-output-trees/master/landscape.svg
   :target: https://landscape.io/github/arjoly/random-output-trees/master
   :alt: Code Health


Random output trees is a python package to grow decision tree ensemble on
randomized output space. The core tree implementation is based on scikit-learn
0.15.2. All provided estimators and transformers are scikit-learn compatible.

If you use this package, please cite

  Joly, A., Geurts, P., & Wehenkel, L. (2014). Random forests with random
  projections of the output space for high dimensional multi-label
  classification.

  ECML-PKDD 2014, Nancy, France


The paper is avaiblable at http://orbi.ulg.ac.be/handle/2268/172146.

Documentation
-------------

The documentation is available at http://arjoly.github.io/random-output-trees/


Dependencies
------------

The required dependencies to build the software are Python >= 2.7,
NumPy >= 1.6.2, SciPy >= 0.9, scikit-learn>=0.15.2 and a working C/C++
compiler.

For running the examples Matplotlib >= 1.1.1 is required and for running the
tests you need nose >= 1.1.2.

For making the documentation, Sphinx==1.2.2 and sphinx-bootstrap-theme==0.4.0
are needed.


Install
-------

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

  python setup.py install --user

To install for all users on Unix/Linux::

  python setup.py build
  sudo python setup.py install


Development
-----------

You can check the latest sources with the command::

    git clone https://github.com/arjoly/random-output-trees

or if you have write privileges::

    git@github.com:arjoly/random-output-trees.git

After installation, you can launch the test suite from outside the
source directory (you will need to have the ``nose`` package installed)::

   $ nosetests -v random_output_trees


Licenses
--------

Copyright (c) 2014, Arnaud Joly. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
