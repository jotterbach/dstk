import pytest
from sklearn import datasets as ds

from DSTK.GAM.gam import GAM, ShapeFunction
import numpy as np

cancer_ds = ds.load_breast_cancer()
data = cancer_ds['data'][:, :20]
labels = 2 * cancer_ds['target'] - 1


def test_gam_training():
    gam = GAM(max_depth=3, max_leaf_nodes=5, random_state=42, balancer_seed=42)
    gam.train(data, labels, n_iter=5, learning_rate=0.0025, num_bags=1, num_workers=3)

    assert_scores = [
        [0.5538394842641805, 0.44616051573581944],
        [0.49861290044203543, 0.5013870995579646],
        [0.5470227126670573, 0.4529772873329428],
        [0.513940794277825, 0.48605920572217504],
        [0.529758125364891, 0.470241874635109]
    ]

    for idx, vec in enumerate(data[:5, :]):
        gam_scores = gam.score(vec)
        np.testing.assert_almost_equal(np.sum(gam_scores), 1.0, 10)
        np.testing.assert_almost_equal(gam_scores, assert_scores[idx], 10)


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


def test_correct_shape_addition():
    func1 = ShapeFunction(_create_partition(np.linspace(1, 10, 10)), np.linspace(1, 10, 11), 'attr_1')
    func2 = func1.add(func1).multiply(0.5)
    assert func1.equals(func2)


def _create_partition(lst_of_splits):
    return np.append(lst_of_splits, np.PINF)


if __name__=='__main__':
    test_gam_training()
    test_returning_value()
    test_func_add()
    test_func_multiply()
    test_func_add_with_equal_splits()
    test_pseudo_response()
