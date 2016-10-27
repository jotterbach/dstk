import sklearn.datasets as ds
from DSTK.FeatureBinning.binners import ConditionalInferenceBinner
import numpy as np
import pandas as pd

cancer_ds = ds.load_breast_cancer()
cancer_data = cancer_ds['data']
cancer_target = cancer_ds['target']

cancer_df = pd.DataFrame(cancer_data, columns=cancer_ds['feature_names'])


def test_recursion():
    col = 'mean radius'
    data = cancer_df[col].values
    cib = ConditionalInferenceBinner('test_dim_{}'.format(col), alpha=0.05)
    cib.fit(data, cancer_target)
    assert cib.cond_proba_buckets == [
        ((np.NINF, 11.75), [0.02, 0.97999999999999998]),
        ((11.75, 13.079999923706055), [0.086956521739130432, 0.91304347826086951]),
        ((13.079999923706055, 15.039999961853027), [0.2878787878787879, 0.71212121212121215]),
        ((15.039999961853027, 16.84000015258789), [0.81481481481481477, 0.18518518518518517]),
        ((16.84000015258789, np.PINF), [0.99152542372881358, 0.0084745762711864406]),
        (np.NaN, [0.37258347978910367, 0.62741652021089633])
    ]


def test_recursion_with_nan():
    col = 'mean area'
    data = cancer_df[col].values
    rand_idx = np.linspace(1, 500, 23).astype(int)
    data[rand_idx] = np.NaN

    cib = ConditionalInferenceBinner('test_dim_{}'.format(col), alpha=0.05)
    cib.fit(data, cancer_target)

    assert cib.cond_proba_buckets == [
        ((np.NINF, 471.29998779296875), [0.030769230769230771, 0.96923076923076923]),
        ((471.29998779296875, 555.0999755859375), [0.13414634146341464, 0.86585365853658536]),
        ((555.0999755859375, 693.7000122070312), [0.31730769230769229, 0.68269230769230771]),
        ((693.7000122070312, 880.2000122070312), [0.83333333333333337, 0.16666666666666666]),
        ((880.2000122070312, np.PINF), [0.99145299145299148, 0.0085470085470085479]),
        (np.NaN, [0.2608695652173913, 0.73913043478260865])
    ]

