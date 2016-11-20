from __future__ import division

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


def test_sgdbolasso_with_cv():
    b = bs.SGDBolasso(bootstrap_fraction=0.5, random_state=42, random_seed=13)

    means = cancer_df.as_matrix().mean(axis=0)
    std = cancer_df.as_matrix().std(axis=0)
    scaled_data = pd.DataFrame((cancer_df.as_matrix() - means) / std, columns=cancer_df.columns)

    cv_params = {'alpha': np.logspace(-3, 2, 10),
                 'n_iter': np.arange(5, 15, 1),
                 'eta0': np.arange(0.1, 1.1, 0.1)}

    estim, rscv = b.fit_cv(scaled_data, targets, cv_params=cv_params, epochs=1000, cv=2, verbose=0, n_jobs=4)

    np.testing.assert_almost_equal(rscv.best_score_, [0.95079086116], decimal=6)

    assert_dict = {'alpha': 0.001, 'eta0': 0.10000000000000001, 'n_iter': 5}
    for key, val in assert_dict.iteritems():
        np.testing.assert_almost_equal(rscv.best_params_[key], val)

    stats_df = estim.get_feature_stats()
    assert stats_df[stats_df.frac_occurence == 1].index.tolist() == ['area error',
                                                                     'radius error',
                                                                     'worst symmetry',
                                                                     'worst radius',
                                                                     'worst perimeter',
                                                                     'worst concave points',
                                                                     'worst area',
                                                                     'worst texture',
                                                                     'mean concave points']

    np.testing.assert_almost_equal(stats_df[stats_df.frac_occurence == 1].coef_mean.values,
                                   [-9.49728727, -12.17192604, -9.11384232, -10.06477512, -8.41949524,
                                    -10.16794578, -8.9595186, -12.84921728, -8.13303665])
