import pandas as pd
import sklearn.datasets as ds
import DSTK.BoostedFeatureSelectors.boselector as bs
import numpy as np

cancer_ds = ds.load_breast_cancer()
cancer_df = pd.DataFrame(cancer_ds['data'], columns=cancer_ds['feature_names'])
targets = pd.Series(cancer_ds['target'])


def test_bolasso():
    b = bs.Bolasso(bootstrap_fraction=0.5, Cs=np.logspace(-1, 1, 3), random_state=42, random_seed=13)
    b.fit(cancer_df, targets, epochs=5)

    assert b.get_feature_stats().columns.tolist() == ['coef_mean', 'coef_std', 'frac_occurence', 'num_occurence']
    assert b.coeff_df.shape == (5, 30)
    assert b.coeff_df.isnull().any().sum() == False


def test_botree():
    b = bs.Botree(bootstrap_fraction=0.5, random_state=42, random_seed=13)
    b.fit(cancer_df, targets, epochs=5)

    assert b.get_feature_stats().columns.tolist() == ['coef_mean', 'coef_std', 'frac_occurence', 'num_occurence']
    assert b.coeff_df.shape == (5, 30)
    assert b.coeff_df.isnull().any().sum() == False


def test_boforest():
    b = bs.Boforest(bootstrap_fraction=0.5, random_state=42, random_seed=13)
    b.fit(cancer_df, targets, epochs=5)

    assert b.get_feature_stats().columns.tolist() == ['coef_mean', 'coef_std', 'frac_occurence', 'num_occurence']
    assert b.coeff_df.shape == (5, 30)
    assert b.coeff_df.isnull().any().sum() == False


def test_botree_2():
    b = bs.Botree(bootstrap_fraction=0.5, calculate_feature_importance=True, feature_importance_threshold=0.25)
    b.fit(cancer_df, targets, epochs=5)

    import DSTK.utils.sampling_helpers as sh

    print b._get_metrics(cancer_df.as_matrix(), targets.values)
    print b._get_metrics(sh.permute_column_of_numpy_array(cancer_df.as_matrix(), 1), targets.values)
#
#     print b.coeff_df['worst symmetry__accuracy']
#     # assert b.get_feature_stats().columns.tolist() == ['coef_mean', 'coef_std', 'frac_occurence', 'num_occurence']
#     # assert b.coeff_df.shape == (5, 30)
#     # assert b.coeff_df.isnull().any().sum() == False