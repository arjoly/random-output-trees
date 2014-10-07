import os
import shutil
import tempfile

from sklearn.utils.testing import with_setup
from sklearn.utils.testing import assert_equal

from random_output_trees.datasets import fetch_drug_interaction


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


@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_drug_interaction():
    dataset = fetch_drug_interaction(tmpdir)

    assert_equal(dataset.data.shape, (1862, 660))
    assert_equal(dataset.target.shape, (1862, 1554))
    assert_equal(len(dataset.feature_names), (660))
    assert_equal(len(dataset.target_names), (1554))
