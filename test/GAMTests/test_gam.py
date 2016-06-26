from sklearn import datasets as ds
from GAM.gam import GAM, ShapeFunction
import numpy as np

cancer_ds = ds.load_breast_cancer()
data = cancer_ds['data'][:, :20]
labels = 2 * cancer_ds['target'] - 1


def test():
    gam = GAM()
    # feat_shapes = gam._calculate_feat_shapes(data, labels)
    gam.train(data, labels, n_iter=105, display_step=5, leaning_rate=0.0025)
    # print 'feature_shape: ', feat_shapes
    for vec in data:
        print gam.score(vec)


def test_returning_value():
    func = gam.ShapeFunction(_create_partition(np.linspace(1, 10, 10)), -1 * np.linspace(1, 10, 11))

    assert func.get_value((10.45)) == -10.0
    assert func.get_value((0.45)) == -1.0
    assert func.get_value((2.0)) == -2.8


def test_func_add():
    func1 = ShapeFunction(np.linspace(1, 10, 10), np.linspace(1, 10, 10))
    func2 = ShapeFunction(np.linspace(1.5, 10.5, 5), -1 * np.linspace(1, 10, 5))
    # print func1, func2
    print func1.add(func2)


def test_func_add_with_infty():

    print np.isposinf(np.PINF)

    func1 = gam.ShapeFunction(_create_partition(np.linspace(1, 10, 10)), np.linspace(1, 10, 11))
    func2 = gam.ShapeFunction(_create_partition(np.linspace(1.5, 10.5, 5)), -1 * np.linspace(1, 10, 6))
    # print func1, func2
    print func1.add(func2)


def test_func_add_with_equal_splits():
    print np.isposinf(np.PINF)

    func1 = gam.ShapeFunction(_create_partition(np.linspace(1, 10, 10)), np.linspace(1, 10, 11))
    func2 = gam.ShapeFunction(_create_partition(np.linspace(1, 10, 10)), -1 * np.linspace(1, 10, 11))
    print func1, func2
    print func1.add(func2)


def _create_partition(lst_of_splits):
    return np.append(lst_of_splits, np.PINF)


def is_sorted(lst):
    return all(lst[i] <= lst[i + 1] for i in xrange(len(lst) - 1))


if __name__=='__main__':
    test()
    # test_returning_value()
    # test_func_add()
    # test_func_add_with_infty()
    # test_func_add_with_equal_splits()