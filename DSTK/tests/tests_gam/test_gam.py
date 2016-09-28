from sklearn import datasets as ds

from DSTK.GAM.gam import GAM, ShapeFunction
from DSTK.tests.tests_gam.test_shape_function import _create_partition
from DSTK.GAM.base_gam import load_from_tar
import numpy as np
import os
import shutil

cancer_ds = ds.load_breast_cancer()
data = cancer_ds['data'][:, :20]
labels = 2 * cancer_ds['target'] - 1

assert_scores = [
    [0.5538394842641805, 0.44616051573581944],
    [0.49861290044203543, 0.5013870995579646],
    [0.5470227126670573, 0.4529772873329428],
    [0.513940794277825, 0.48605920572217504],
    [0.529758125364891, 0.470241874635109]
]


test_root_folder = '/tmp/test_gam_serialization'

def teardown():
    if os.path.exists(test_root_folder):
        shutil.rmtree(test_root_folder)

def test_gam_training():
    gam = GAM(max_depth=3, max_leaf_nodes=5, random_state=42, balancer_seed=42)
    gam.train(data, labels, n_iter=5, learning_rate=0.0025, num_bags=1, num_workers=3)

    for idx, vec in enumerate(data[:5, :]):
        gam_scores = gam.score(vec)
        np.testing.assert_almost_equal(np.sum(gam_scores), 1.0, 10)
        np.testing.assert_almost_equal(gam_scores, assert_scores[idx], 10)


def test_correct_scoring():
    func1 = ShapeFunction(_create_partition(np.linspace(1, 10, 10)), np.linspace(1, 10, 11), 'attr_1')
    func2 = ShapeFunction(_create_partition(np.linspace(1, 10, 10)), -1 * np.linspace(1, 10, 11), 'attr_2')

    gam = GAM(max_depth=3, max_leaf_nodes=5, random_state=42)
    gam.shapes = {'attr_1': func1,
                  'attr_2': func2}
    gam.feature_names = ['attr_1', 'attr_2']
    data = np.asarray([[1, 1], [-1, -1]])

    for vec in data:
        # the shape function cancel each other in their effect
        # hence the exponent is zero and the probability 0.5
        assert gam.score(vec) == [0.5, 0.5]


def test_correct_scoring_2():
    func1 = ShapeFunction(_create_partition(np.linspace(1, 10, 10)), 0.75 * np.ones(11), 'attr_1')
    func2 = ShapeFunction(_create_partition(np.linspace(1, 10, 10)),  -0.25 * np.ones(11), 'attr_2')

    gam = GAM(max_depth=3, max_leaf_nodes=5, random_state=42)
    gam.shapes = {'attr_1': func1,
                  'attr_2': func2}
    gam.feature_names = ['attr_1', 'attr_2']
    data = np.asarray([[1, 1], [-1, -1]])

    for vec in data:
        # the shape function cancel each other in their effect
        # hence the exponent is zero and the probability 0.5
        assert gam.logit_score(vec) == 0.5
        assert gam.score(vec) == [0.2689414213699951, 0.7310585786300049]


def test_pseudo_response():

    func1 = ShapeFunction(_create_partition([0]), [np.log(3), np.log(100)], 'attr_1')

    gam = GAM(max_depth=3, max_leaf_nodes=5, random_state=42)
    gam.shapes = {'attr_1': func1}
    gam.feature_names = ['attr_1']

    data = np.asarray([[-1], [1]])
    pseudo_resp = gam._get_pseudo_responses(data, [1, -1])
    np.testing.assert_almost_equal(pseudo_resp, [0.2, -1.9998],
                                   decimal=6,
                                   verbose=True,
                                   err_msg="Pseudo Response doesn't match")


def test_serialization_deserialization():
    gam = GAM(max_depth=3, max_leaf_nodes=5, random_state=42, balancer_seed=42)
    gam.train(data, labels, n_iter=5, learning_rate=0.0025, num_bags=1, num_workers=3)

    gam.serialize('gbt_gam', file_path='/tmp/test_gam_serialization')

    scoring_gam = load_from_tar('/tmp/test_gam_serialization/gbt_gam.tar.gz')

    for idx, vec in enumerate(data[:5, :]):
        gam_scores = scoring_gam.score(vec)
        np.testing.assert_almost_equal(np.sum(gam_scores), 1.0, 10)
        np.testing.assert_almost_equal(gam_scores, assert_scores[idx], 10)

