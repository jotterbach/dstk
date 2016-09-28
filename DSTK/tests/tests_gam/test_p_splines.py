import pytest
import numpy as np
import DSTK.GAM.utils.p_splines as ps
import DSTK.GAM.PSplineGAM as psgam
import sklearn.datasets as ds
import pandas as pd

cancer_ds = ds.load_breast_cancer()
data = cancer_ds['data']
target = cancer_ds['target']

data_df = pd.DataFrame(data, columns=cancer_ds['feature_names'])


def test_psline_smoothing():
    penalties = [0.0, 1.0, 10.0]

    spl = ps.PSpline()

    gcv_error = [spl.fit(data[:, 0].tolist(), target.tolist(), penalty=penalty).gcv_score() for penalty in penalties]
    opt_lambda = penalties[np.argmin(gcv_error)]
    spl.fit(data[:, 0].tolist(), target.tolist(), penalty=opt_lambda)
    np.testing.assert_array_almost_equal(spl.predict(data[:5, 0]), [0.01498067, 0.00344838, 0.00582068, 0.95736314, 0.0048818], 6)


def test_get_basis_matrix_for_array():
    gam = psgam.PSplineGAM(num_percentiles=5)

    arr = np.random.uniform(size=(100, 2))

    gam._create_knots_dict(arr, gam.num_percentiles)

    knots0 = ps._get_percentiles(arr[:, 0], num_percentiles=5)
    knots1 = ps._get_percentiles(arr[:, 1], num_percentiles=5)

    basis = gam._get_basis_for_array(arr)

    np.testing.assert_array_equal(basis[:, :, 0], ps._get_basis_vector(arr[:, 0], knots0, with_intercept=False))
    np.testing.assert_array_equal(basis[:, :, 1], ps._get_basis_vector(arr[:, 1], knots1, with_intercept=False))


def test_flatten_basis_matrix_for_regression():
    gam = psgam.PSplineGAM(num_percentiles=2)

    arr = np.random.uniform(size=(4, 3))

    gam._create_knots_dict(arr, gam.num_percentiles)

    knots0 = ps._get_percentiles(arr[:, 0], num_percentiles=2)
    knots1 = ps._get_percentiles(arr[:, 1], num_percentiles=2)
    knots2 = ps._get_percentiles(arr[:, 2], num_percentiles=2)

    dim0_basis = ps._get_basis_vector(arr[:, 0], knots0, with_intercept=False)
    dim1_basis = ps._get_basis_vector(arr[:, 1], knots1, with_intercept=False)
    dim2_basis = ps._get_basis_vector(arr[:, 2], knots2, with_intercept=False)

    manual_concat = np.asarray([np.asarray([1.0] + vec_0.tolist() + vec_1.tolist() + vec_2.tolist()) for vec_0, vec_1, vec_2 in zip(dim0_basis, dim1_basis, dim2_basis)])

    base_set = gam._flatten_basis_for_fitting(arr)

    np.testing.assert_array_equal(base_set, manual_concat)


def test_p_spline_fitting():
    spline_fitter = psgam.PSplineGAM()

    spline_fitter.train(data, target)

    np.testing.assert_array_almost_equal(
        spline_fitter.predict(data[:10, :]),
        [0.38021172202232384, 0.077934743087122921, 0.29684891104959737,
         0.22027919416405917, 0.60467894303470937, 0.12925950800213407,
         0.33262495655469365, 0.24803627163131964, 0.15091752319056068,
         0.14379721963252465],
        4)


def test_p_spline_fitting_with_dataframe():
    spline_fitter = psgam.PSplineGAM()

    spline_fitter.train(data_df, target)

    assert spline_fitter.feature_names == data_df.columns.tolist()
    assert set(spline_fitter.shapes.keys()) == set(data_df.columns.tolist())