import pytest
import numpy as np
import DSTK.GAM.utils.p_splines as ps
import DSTK.GAM.PSplineGAM as psgam
import sklearn.datasets as ds

cancer_ds = ds.load_breast_cancer()
data = cancer_ds['data']
target = cancer_ds['target']


def test_psline_smoothing():
    penalties = [0.0, 1.0, 10.0]

    spl = ps.PSpline()

    gcv_error = [spl.fit(data[:, 0].tolist(), target.tolist(), penalty=penalty).gcv_score() for penalty in penalties]
    opt_lambda = penalties[np.argmin(gcv_error)]
    # print gcv_error, opt_lambda
    spl.fit(data[:, 0].tolist(), target.tolist(), penalty=opt_lambda)

#
# def test_get_basis_matrix_for_array():
#     arr = np.random.uniform(size=(100, 2))
#     knots0 = ps._get_percentiles(arr[:, 0], num_percentiles=5)
#     knots1 = ps._get_percentiles(arr[:, 1], num_percentiles=5)
#
#     basis = ps._get_basis_for_array(arr, num_percentiles=5, with_intercept=True)
#
#     np.testing.assert_array_equal(basis[:, :, 0], ps._get_basis_vector(arr[:, 0], knots0))
#     np.testing.assert_array_equal(basis[:, :, 1], ps._get_basis_vector(arr[:, 1], knots1))
#
#
# def test_flatten_basis_matrix_for_regression():
#     arr = np.random.uniform(size=(4, 3))
#     knots0 = ps._get_percentiles(arr[:, 0], num_percentiles=2)
#     knots1 = ps._get_percentiles(arr[:, 1], num_percentiles=2)
#     knots2 = ps._get_percentiles(arr[:, 2], num_percentiles=2)
#
#     dim0_basis = ps._get_basis_vector(arr[:, 0], knots0, with_intercept=False)
#     dim1_basis = ps._get_basis_vector(arr[:, 1], knots1, with_intercept=False)
#     dim2_basis = ps._get_basis_vector(arr[:, 2], knots2, with_intercept=False)
#
#     manual_concat = np.asarray([np.asarray([1.0] + vec_0.tolist() + vec_1.tolist() + vec_2.tolist()) for vec_0, vec_1, vec_2 in zip(dim0_basis, dim1_basis, dim2_basis)])
#
#     base_set = ps._flatten_basis_for_fitting(arr, num_percentiles=2)
#
#     print base_set.shape
#
#     np.testing.assert_array_equal(base_set, manual_concat)


def test_p_spline_fitting():
    spline_fitter = psgam.PSplineGAM()

    filter_data = data[:, :10]
    print filter_data.shape

    spline_fitter.fit(filter_data, target, )

    print spline_fitter._get_shape(1, filter_data[:, 1]).shape
