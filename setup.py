#! /usr/bin/env python
#
# Author : Arnaud Joly
#
# License: BSD 3 clause

import sys
import os
import shutil
from distutils.command.clean import clean as Clean

DISTNAME = 'random-output-trees'
DESCRIPTION = "High dimension output tree classifier and regressor"
LONG_DESCRIPTION = open('README.rst').read()
MAINTAINER = 'Arnaud Joly'
MAINTAINER_EMAIL = 'arnaud.v.joly@gmail.com'
URL = 'http://arjoly.github.io/random-output-trees/'
LICENSE = 'BSD'
DOWNLOAD_URL = 'https://github.com/arjoly/random-output-trees/archive/master.zip'
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

import random_output_trees
VERSION = random_output_trees.__version__

import setuptools  # we are using a setuptools namespace
from numpy.distutils.core import setup

class CleanCommand(Clean):
    description = "Remove build directories, and compiled file in the source tree"

    def run(self):
        Clean.run(self)
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('random_output_trees'):
            for filename in filenames:
                if (filename.endswith('.so') or filename.endswith('.pyd')
                             or filename.endswith('.dll')
                             or filename.endswith('.pyc')):
                    os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    config.add_subpackage('random_output_trees')

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
          classifiers=CLASSIFIERS,
          cmdclass={'clean': CleanCommand},
    )
