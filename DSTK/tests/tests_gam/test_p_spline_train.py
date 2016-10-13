import numpy as np
import DSTK.GAM.PSplineGAM as psgam
import sklearn.datasets as ds
import pandas as pd

cancer_ds = ds.load_breast_cancer()
data = cancer_ds['data']
target = cancer_ds['target']

data_df = pd.DataFrame(data, columns=cancer_ds['feature_names'])

# TODO: This test fails on TravisCI
# possibly connected to statsmodel being compiled differently
# solution for now is to delete this file before executing tox on TravisCI
def test_p_spline_fitting():
    spline_fitter = psgam.PSplineGAM(max_iter=10)

    spline_fitter.train(data, target)

    np.testing.assert_array_almost_equal(
        spline_fitter.predict(data[:10, :]),
        [0.38010625884484806, 0.078015764411492383, 0.29677559014935362,
         0.21998755394156164, 0.60436774552322059, 0.12905647770342621,
         0.33253440991848549, 0.24778384843818288, 0.15079040243134087,
         0.14349952871877761],
        4)
