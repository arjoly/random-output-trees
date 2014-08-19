#! /usr/bin/env python
#
# Copyright (C) 2012 Arnaud Joly

import sys
import os

DISTNAME = 'randomized-output-forest'
DESCRIPTION = "High dimension output tree classifier and regressor"
LONG_DESCRIPTION = open('README.rst').read()
MAINTAINER = 'Arnaud Joly'
MAINTAINER_EMAIL = 'arnaud.v.joly@gmail.com'
URL = 'TODO'
LICENSE = 'TODO' #TODO switch to new bsd later
DOWNLOAD_URL = 'TODO'
CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved',
    'Programming Language :: C',
    'Programming Language :: Python',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS'
]

import randomized_output_forest
VERSION = randomized_output_forest.__version__


import setuptools  # we are using a setuptools namespace
from numpy.distutils.core import setup

def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    config.add_subpackage('randomized_output_forest')

    return config

if __name__ == "__main__":

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    setup(configuration=configuration,
          name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          zip_safe=False, # the package can run out of an .egg file
          classifiers=CLASSIFIERS
    )
