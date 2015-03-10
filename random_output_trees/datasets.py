"""

Module for datasets loading and fetchers.

"""

from __future__ import division, print_function, absolute_import

import os
from functools import partial

import shutil
import tarfile

try:
    # Python 2
    from urllib2 import HTTPError
    from urllib2 import quote
    from urllib2 import urlopen
except ImportError:
    # Python 3+
    from urllib.error import HTTPError
    from urllib.parse import quote
    from urllib.request import urlopen


import numpy as np

from sklearn.datasets import get_data_home
from sklearn.datasets.base import Bunch


__all__ = [
    "fetch_drug_interaction",
    "fetch_protein_interaction",
]


def _fetch_drug_protein(data_home=None):
    """Fetch drug-protein dataset from the server"""

    base_url = "http://cbio.ensmp.fr/~yyamanishi/substr-domain/"

    # check if this data set has been already downloaded
    data_home = get_data_home(data_home)
    data_home = os.path.join(data_home, 'drug-protein')
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    for base_name in ["drug_repmat.txt", "target_repmat.txt",
                      "inter_admat.txt"]:
        filename = os.path.join(data_home, base_name)

        if not os.path.exists(filename):
            urlname = base_url + base_name

            print("Download data at {}".format(urlname))

            try:
                url = urlopen(urlname)
            except HTTPError as e:
                if e.code == 404:
                    e.msg = "Dataset drug-protein '%s' not found." % base_name
                raise

            try:
                with open(filename, 'w+b') as fhandle:
                    shutil.copyfileobj(url, fhandle)
            except:
                os.remove(filename)
                raise

            url.close()

    return data_home


def fetch_drug_interaction(data_home=None):
    """Fetch the drug-interaction dataset

    Constant features were removed.

    =========================== ===================================
    Domain                         drug-protein interaction network
    Features                                   Biological (see [1])
    output                                      interaction network
    Drug matrix                    (sample, features) = (1862, 660)
    Newtork interaction matrix     (samples, labels) = (1862, 1554)
    =========================== ===================================


    Parameters
    ----------
    data_home: optional, default: None
        Specify another download and cache folder for the data sets. By default
        all scikit learn data is stored in '~/scikit_learn_data' subfolders.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels,
        'target_names', the original names of the target columns and
        'feature_names', the original names of the dataset columns.

    References
    ----------
    .. [1] Yamanishi, Y., Pauwels, E., Saigo, H., & Stoven, V. (2011).
           Extracting sets of chemical substructures and protein domains
           governing drug-target interactions. Journal of chemical information
           and modeling, 51(5), 1183-1194.

    """
    data_home = _fetch_drug_protein(data_home=data_home)

    drug_fname = os.path.join(data_home, "drug_repmat.txt")
    data = np.loadtxt(drug_fname, dtype=float, skiprows=1)
    data = data[:, 1:]  # skip id column
    mask_constant = np.var(data, axis=0) != 0.
    data = data[:, mask_constant]  # remove constant columns

    with open(drug_fname, 'r') as fhandle:
        feature_names = fhandle.readline().split("\t")
        feature_names = np.array(feature_names)[mask_constant].tolist()

    interaction_fname = os.path.join(data_home, "inter_admat.txt")
    target = np.loadtxt(interaction_fname, dtype=float, skiprows=1)
    target = target[:, 1:]  # skip id column
    with open(interaction_fname, 'r') as fhandle:
        target_names = fhandle.readline().split("\t")

    return Bunch(data=data, target=target, feature_names=feature_names,
                 target_names=target_names)


def fetch_protein_interaction(data_home=None):
    """Fetch the protein-interaction dataset

    Constant features were removed

    =========================== ===================================
    Domain                         drug-protein interaction network
    Features                                   Biological (see [1])
    output                                      interaction network
    Drug matrix                    (sample, features) = (1554, 876)
    Newtork interaction matrix     (samples, labels) = (1554, 1862)
    =========================== ===================================

    Parameters
    ----------
    data_home: optional, default: None
        Specify another download and cache folder for the data sets. By default
        all scikit learn data is stored in '~/scikit_learn_data' subfolders.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels and
        'feature_names', the original names of the dataset columns.

    References
    ----------
    .. [1] Yamanishi, Y., Pauwels, E., Saigo, H., & Stoven, V. (2011).
           Extracting sets of chemical substructures and protein domains
           governing drug-target interactions. Journal of chemical information
           and modeling, 51(5), 1183-1194.

    """
    data_home = _fetch_drug_protein(data_home=data_home)

    protein_fname = os.path.join(data_home, "target_repmat.txt")
    data = np.loadtxt(protein_fname, dtype=float, skiprows=1,
                      usecols=range(1, 877))  # skip id column

    mask_constant = np.var(data, axis=0) != 0.
    data = data[:, mask_constant]   # remove constant columns

    with open(protein_fname, 'r') as fhandle:
        feature_names = fhandle.readline().split("\t")
        feature_names = np.array(feature_names)[mask_constant].tolist()

    interaction_fname = os.path.join(data_home, "inter_admat.txt")
    target = np.loadtxt(interaction_fname, dtype=float, skiprows=1)
    target = target[:, 1:]  # skip id column
    target = target.T

    return Bunch(data=data, target=target, feature_names=feature_names)
