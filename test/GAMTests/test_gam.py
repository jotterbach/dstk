from sklearn import datasets as ds
from GAM.gam import GAM, ShapeFunction
import numpy as np

cancer_ds = ds.load_breast_cancer()
data = cancer_ds['data'][:, :20]
labels = 2 * cancer_ds['target'] - 1


def test_gam_training():
    gam = GAM(max_depth=3, max_leaf_nodes=5, random_state=42)
    gam.train(data, labels, n_iter=5, display_step=5, leaning_rate=0.0025)

    assert_scores = [
        (0.51027618321245982, 0.48972381678754023),
        (0.49717529637181201, 0.50282470362818799),
        (0.51032308029509865, 0.4896769197049014),
        (0.49361964153418664, 0.50638035846581331),
        (0.50655527797695332, 0.49344472202304668),
    ]

    for idx, vec in enumerate(data[:5, :]):
        assert gam.score(vec) == assert_scores[idx]


def test_returning_value():
    func = ShapeFunction(_create_partition(np.linspace(1, 10, 10)), -1 * np.linspace(1, 10, 11))

    assert func.get_value((10.45)) == -10.0
    assert func.get_value((0.45)) == -1.0
    assert func.get_value((2.0)) == -2.8


def test_func_multiply():
    func1 = ShapeFunction(np.linspace(1, 3, 3), np.linspace(1, 3, 3))
    func2 = ShapeFunction(np.linspace(1, 3, 3), 0.5 * np.linspace(1, 3, 3))

    assert func1.multiply(0.5).equals(func2)


def test_func_add():
    func1 = ShapeFunction(np.linspace(1, 3, 3), np.linspace(1, 3, 3))
    func2 = ShapeFunction(np.linspace(1.5, 10.5, 5), -1 * np.linspace(1, 10, 5))

    assert func1 != func2

    func3 = ShapeFunction([1.0, 1.5, 2.0, 3.0, 3.75, 6.0, 8.25, 10.5],
                          [1.0, -1.0, 2.0, 3.0, -3.25, -5.5, -7.75, -10.0])

    assert func1.add(func2).equals(func3)


def test_func_add_with_equal_splits():

    func1 = ShapeFunction(_create_partition(np.linspace(1, 10, 10)), np.linspace(1, 10, 11))
    func2 = ShapeFunction(_create_partition(np.linspace(1, 10, 10)), -1 * np.linspace(1, 10, 11))
    func3 = ShapeFunction(_create_partition(np.linspace(1, 10, 10)), np.zeros(11))
    assert func1.add(func2).equals(func3)


def _create_partition(lst_of_splits):
    return np.append(lst_of_splits, np.PINF)


if __name__=='__main__':
    test_gam_training()
    test_returning_value()
    test_func_add()
    test_func_multiply()
    test_func_add_with_equal_splits()