import pandas as pd
import sklearn.datasets as ds
import Bolasso.bolasso as bl
import numpy as np

cancer_ds = ds.load_breast_cancer()
cancer_df = pd.DataFrame(cancer_ds['data'], columns=cancer_ds['feature_names'])
targets = pd.Series(cancer_ds['target'])


def test_bolasso():
    b = bl.Bolasso(bootstrap_fraction=0.5, Cs=np.logspace(-1, 1, 3), random_state=42, random_seed=13)
    b.fit(cancer_df, targets, epochs=5)

    assert b.get_feature_stats().columns.tolist() == ['coef_mean', 'coef_std', 'frac_occurence', 'num_occurence']
    assert b.bolasso_df.shape == (5, 30)
    assert b.bolasso_df.isnull().any().sum() == False

