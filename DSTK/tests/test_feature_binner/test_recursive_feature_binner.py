import sklearn.datasets as ds
import numpy as np
from DSTK.FeatureBinning import binners as tfb

cancer_ds = ds.load_breast_cancer()
data = cancer_ds['data']
target = cancer_ds['target']


def test_recursion():
    binner = tfb.TreeBasedFeatureBinning('test', max_leaf_nodes=4)
    binner.fit(data[:, 0], target)

    assert binner._count_buckets == [
        ((np.NINF, 13.094999313354492), [13.0, 252.0]),
        ((13.094999313354492, 15.045000076293945), [38.0, 94.0]),
        ((15.045000076293945, 16.924999237060547), [44.0, 10.0]),
        ((16.924999237060547, np.PINF), [117.0, 1.0])]
    assert binner.cond_proba_buckets == [
        ((np.NINF, 13.094999313354492), [0.04905660377358491, 0.9509433962264151]),
         ((13.094999313354492, 15.045000076293945), [0.2878787878787879, 0.7121212121212122]),
         ((15.045000076293945, 16.924999237060547), [0.8148148148148148, 0.18518518518518517]),
         ((16.924999237060547, np.PINF), [0.9915254237288136, 0.00847457627118644]),
         (np.NaN, [0.5, 0.5])]


def test_recursion_with_mdlp():
    binner = tfb.TreeBasedFeatureBinning('test', mdlp=True)
    binner.fit(data[:, 0], target)

    assert binner._count_buckets == [
        ((np.NINF, 13.094999313354492), [13.0, 252.0]),
        ((13.094999313354492, 15.045000076293945), [38.0, 94.0]),
        ((15.045000076293945, 17.880001068115234), [64.0, 11.0]),
        ((17.880001068115234, np.PINF), [97.0, 0.0])]

    assert binner.cond_proba_buckets == [
        ((np.NINF, 13.094999313354492), [0.04905660377358491, 0.9509433962264151]),
        ((13.094999313354492, 15.045000076293945), [0.2878787878787879, 0.7121212121212122]),
        ((15.045000076293945, 17.880001068115234), [0.8533333333333334, 0.14666666666666667]),
        ((17.880001068115234, np.PINF), [1.0, 0.0]),
        (np.NaN, [0.5, 0.5])]


def test_fit():
    feats = [0, 1, 2, np.nan, 5, np.nan]
    labels = [0, 1, 0, 1, 1, 0]

    binner = tfb.TreeBasedFeatureBinning('test', max_leaf_nodes=3)
    binner.fit(feats, labels)
    assert binner.cond_proba_buckets == [
        ((np.NINF, 0.5), [1.0, 0.0]),
        ((0.5, 1.5), [0.0, 1.0]),
        ((1.5, np.PINF), [0.5, 0.5]),
        (np.NaN, [0.5, 0.5])]


def test_transform():
    feats = [0, 1, 2, np.nan, 5, np.nan]
    labels = [0, 1, 0, 1, 1, 0]

    binner =  tfb.TreeBasedFeatureBinning('test', max_leaf_nodes=3)
    assert binner.fit_transform(feats, labels) == [0.0, 1.0, 0.5, 0.5, 0.5, 0.5]


def test_transform_to_categorical():
    feats = [0, 1, 2, np.nan, 5, np.nan]
    labels = [0, 1, 0, 1, 1, 0]

    binner = tfb.TreeBasedFeatureBinning('test')
    binner.fit(feats, labels)
    cats = binner.transform_to_categorical(feats)

    assert cats == [
        '(-inf, 0.5]',
        '(0.5, 1.5]',
        '(1.5, 3.5]',
        'is_nan',
        '(3.5, inf]',
        'is_nan']


def test_fit_transform():
    feats = [0, 1, 2, np.nan, 5, np.nan]
    labels = [0, 1, 0, 1, 1, 0]

    binner = tfb.TreeBasedFeatureBinning('test', max_leaf_nodes=3)
    trans = binner.fit_transform(feats, labels)
    assert trans == [0.0, 1.0, 0.5, 0.5, 0.5, 0.5]


