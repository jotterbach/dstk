import sklearn.datasets as ds
from DSTK.FeatureBinning.binners import ConditionalInferenceBinner
import numpy as np
import pandas as pd

cancer_ds = ds.load_breast_cancer()
cancer_data = cancer_ds['data']
cancer_target = cancer_ds['target']

cancer_df = pd.DataFrame(cancer_data, columns=cancer_ds['feature_names'])

assert_bins = [((np.NINF, 11.75), [0.02, 0.97999999999999998]),
               ((11.75, 13.079999923706055), [0.086956521739130432, 0.91304347826086951]),
               ((13.079999923706055, 15.039999961853027), [0.2878787878787879, 0.71212121212121215]),
               ((15.039999961853027, 16.84000015258789), [0.81481481481481477, 0.18518518518518517]),
               ((16.84000015258789, np.PINF), [0.99152542372881358, 0.0084745762711864406]),
               (np.NaN, [0.5, 0.5])]


def test_recursion_pandas():
    col = 'mean radius'
    data = cancer_df[col].values
    cib = ConditionalInferenceBinner('test_dim_{}'.format(col), alpha=0.05)
    cib.fit(data, cancer_target)
    assert cib.cond_proba_buckets == assert_bins
