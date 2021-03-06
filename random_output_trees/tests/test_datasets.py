import os
import shutil
import tempfile

from sklearn.utils.testing import with_setup
from sklearn.utils.testing import assert_equal

from random_output_trees.datasets import fetch_drug_interaction
from random_output_trees.datasets import fetch_protein_interaction
from random_output_trees._utils import skipped

tmpdir = None


def setup_tmpdata():
    # create temporary dir
    global tmpdir
    tmpdir = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmpdir, 'drug-protein'))


def teardown_tmpdata():
    # remove temporary dir
    if tmpdir is not None:
        shutil.rmtree(tmpdir)

@skipped
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_drug_protein():
    dataset = fetch_drug_interaction(tmpdir)

    assert_equal(dataset.data.shape, (1862, 660))
    assert_equal(dataset.target.shape, (1862, 1554))
    assert_equal(len(dataset.feature_names), 660)
    assert_equal(len(dataset.target_names), 1554)

    dataset = fetch_protein_interaction(tmpdir)
    assert_equal(dataset.data.shape, (1554, 876))
    assert_equal(dataset.target.shape, (1554, 1862))
    assert_equal(len(dataset.feature_names), 876)
