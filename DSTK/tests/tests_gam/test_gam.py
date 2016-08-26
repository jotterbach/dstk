import pytest
from sklearn import datasets as ds
from collections import Counter

from DSTK.GAM.gam import GAM, ShapeFunction
import numpy as np

cancer_ds = ds.load_breast_cancer()
data = cancer_ds['data'][:, :20]
labels = 2 * cancer_ds['target'] - 1


def test_downsample():
    initial_cntr = Counter(labels)
    min_size = np.min(initial_cntr.values())
    down_data, down_labels = GAM._downsample_majority_class(data, labels)

    for down_idx, vec in enumerate(down_data):
        idx = np.where((data == vec).all(axis=1))[0].squeeze()
        assert labels[idx] == down_labels[down_idx]

    cntr = Counter(down_labels)
    assert cntr.get(1) == min_size
    assert cntr.get(1) == cntr.get(-1)


def test_upsample():
    initial_cntr = Counter(labels)
    max_size = np.max(initial_cntr.values())
    up_data, up_labels = GAM._upsample_minority_class(data, labels)

    for down_idx, vec in enumerate(up_data):
        idx = np.where((data == vec).all(axis=1))[0].squeeze()
        assert labels[idx] == up_labels[down_idx]

    cntr = Counter(up_labels)
    assert cntr.get(1) == max_size
    assert cntr.get(1) == cntr.get(-1)


def test_gam_training():
    gam = GAM(max_depth=3, max_leaf_nodes=5, random_state=42)
    gam.train(data, labels, n_iter=5, display_step=5, learning_rate=0.0025)

    assert_scores = [
        (0.47935468041733226, 0.5206453195826677),
        (0.48710541659624385, 0.5128945834037562),
        (0.48732856271802855, 0.5126714372819714),
        (0.4683372073154978, 0.5316627926845022),
        (0.48593558483691957, 0.5140644151630804)
    ]

    for idx, vec in enumerate(data[:5, :]):
        assert gam.score(vec) == assert_scores[idx]


def test_returning_value():
    func = ShapeFunction(_create_partition(np.linspace(1, 10, 10)), -1 * np.linspace(1, 10, 11), 'test_1')

    assert func.get_value(10.45) == -10.0
    assert func.get_value(0.45) == -1.0
    assert func.get_value(2.0) == -2.8


def test_func_multiply():
    func1 = ShapeFunction(np.linspace(1, 3, 3), np.linspace(1, 3, 3), 'test_1')
    func2 = ShapeFunction(np.linspace(1, 3, 3), 0.5 * np.linspace(1, 3, 3), 'test_2')

    assert func1.multiply(0.5).equals(func2)


def test_func_add():
    func1 = ShapeFunction(np.linspace(1, 3, 3), np.linspace(1, 3, 3), 'test_1')
    func2 = ShapeFunction(np.linspace(1.5, 10.5, 5), -1 * np.linspace(1, 10, 5), 'test_1')

    assert func1 != func2

    func3 = ShapeFunction([1.0, 1.5, 2.0, 3.0, 3.75, 6.0, 8.25, 10.5],
                          [1.0, -1.0, 2.0, 3.0, -3.25, -5.5, -7.75, -10.0], 'test_1')

    assert func1.add(func2).equals(func3)


def test_func_add_2():
    func1 = ShapeFunction([np.PINF], [0], 'test_1')
    func2 = ShapeFunction([0, 1, 2], [-1, 1, -2], 'test_1')

    assert func1 != func2

    func3 = ShapeFunction([0.0, 1.0, 2.0, np.PINF],
                          [-1.0, 1.0, -2.0, 0.0], 'test_1')

    assert func1.add(func2).equals(func3)


def test_func_add_fails_with_different_features():
    func1 = ShapeFunction(np.linspace(1, 3, 3), np.linspace(1, 3, 3), 'test_1')
    func2 = ShapeFunction(np.linspace(1.5, 10.5, 5), -1 * np.linspace(1, 10, 5), 'test_2')

    assert func1 != func2
    with pytest.raises(AssertionError):
        func1.add(func2)


def test_func_add_with_equal_splits():

    func1 = ShapeFunction(_create_partition(np.linspace(1, 10, 10)), np.linspace(1, 10, 11), 'test_1')
    func2 = ShapeFunction(_create_partition(np.linspace(1, 10, 10)), -1 * np.linspace(1, 10, 11), 'test_1')
    func3 = ShapeFunction(_create_partition(np.linspace(1, 10, 10)), np.zeros(11), 'test_1')
    assert func1.add(func2).equals(func3)


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
        assert gam.score(vec) == (0.5, 0.5)


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
        assert gam.score(vec) == (0.2689414213699951, 0.7310585786300049)


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


def _create_partition(lst_of_splits):
    return np.append(lst_of_splits, np.PINF)


if __name__=='__main__':
    test_gam_training()
    test_returning_value()
    test_func_add()
    test_func_multiply()
    test_func_add_with_equal_splits()
    test_pseudo_response()
