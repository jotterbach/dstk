import numpy as np

from DSTK.FeatureBinning._utils import _filter_special_values, _naive_bayes_bins


def test_special_values_filter():
    vals = np.asarray([-1.0, 3.1, 11.3, 0.5, np.NaN, -1.0, 3.1])
    special_vals = np.asarray([-1.0, np.NaN])

    idx_dct = _filter_special_values(vals, special_vals)

    np.testing.assert_equal(idx_dct[str(np.NaN)], [4])
    np.testing.assert_equal(idx_dct[str(-1.0)], [0, 5])
    np.testing.assert_equal(idx_dct['regular'], [1, 2, 3, 6])


def test_special_values_filter_2():
    vals = np.asarray([-1.0, 3.1, 11.3, 0.5, np.NaN, -1.0, 3.1])
    special_vals = np.asarray([np.NaN])

    idx_dct = _filter_special_values(vals, special_vals)

    np.testing.assert_equal(idx_dct[str(np.NaN)], [4])
    np.testing.assert_equal(idx_dct['regular'], [0, 1, 2, 3, 5, 6])


def test_naive_bayes_bin():
    target = np.asarray([0, 1, 1, 1, 0, 1, 0, 1, 1, 1])

    mean = target.mean()
    np.testing.assert_almost_equal(_naive_bayes_bins(target, 0.5, num_classes=2), [1-mean, mean], decimal=10)

    np.testing.assert_almost_equal(_naive_bayes_bins([], 0.6, num_classes=2), [0.4, 0.6], decimal=10)