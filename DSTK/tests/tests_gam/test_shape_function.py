import pytest

from DSTK.GAM.gam import ShapeFunction
import numpy as np


def _create_partition(lst_of_splits):
    return np.append(lst_of_splits, np.PINF)


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


def test_correct_shape_addition():
    func1 = ShapeFunction(_create_partition(np.linspace(1, 10, 10)), np.linspace(1, 10, 11), 'attr_1')
    func2 = func1.add(func1).multiply(0.5)
    assert func1.equals(func2)

