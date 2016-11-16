import sklearn.datasets as ds
import numpy as np
from DSTK.FeatureBinning import decision_tree_binner as tfb

cancer_ds = ds.load_breast_cancer()
data = cancer_ds['data']
target = cancer_ds['target']


def test_recursion():
    binner = tfb.DecisionTreeBinner('test', max_leaf_nodes=4)
    binner.fit(data[:, 0], target)

    np.testing.assert_equal(binner.splits, [13.094999313354492, 15.045000076293945, 16.924999237060547, np.PINF, np.NaN])
    np.testing.assert_equal(binner.values, [[0.04905660377358491, 0.9509433962264151],
                                            [0.2878787878787879, 0.7121212121212122],
                                            [0.8148148148148148, 0.18518518518518517],
                                            [0.9915254237288136, 0.00847457627118644],
                                            [0.37258347978910367, 0.62741652021089633]])


def test_recursion_with_mdlp():
    binner = tfb.DecisionTreeBinner('test', mdlp=True)
    binner.fit(data[:, 0], target)

    np.testing.assert_equal(binner.splits, [13.094999313354492, 15.045000076293945, 17.880001068115234, np.PINF, np.NaN])
    np.testing.assert_equal(binner.values, [[0.04905660377358491, 0.9509433962264151],
                                            [0.2878787878787879, 0.7121212121212122],
                                            [0.8533333333333334, 0.14666666666666667],
                                            [1.0, 0.0],
                                            [0.37258347978910367, 0.62741652021089633]])


def test_str_repr_with_mdlp():

    assert_str = \
    """<= 13.0949993134: [ 0.0490566  0.9509434]
<= 15.0450000763: [ 0.28787879  0.71212121]
<= 17.8800010681: [ 0.85333333  0.14666667]
<= inf: [ 1.  0.]
NaN: [ 0.37258348  0.62741652]"""

    binner = tfb.DecisionTreeBinner('test', mdlp=True)
    binner.fit(data[:, 0], target)

    assert str(binner) == assert_str


def test_fit():
    feats = [0, 1, 2, np.nan, 5, np.nan]
    labels = [0, 1, 0, 1, 1, 0]

    binner = tfb.DecisionTreeBinner('test', max_leaf_nodes=3)
    binner.fit(feats, labels)

    np.testing.assert_equal(binner.splits, [0.5, 1.5, np.PINF, np.NaN])
    np.testing.assert_equal(binner.values, [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.5, 0.5]])


def test_transform():
    feats = [0, 1, 2, np.nan, 5, np.nan]
    labels = [0, 1, 0, 1, 1, 0]

    binner = tfb.DecisionTreeBinner('test', max_leaf_nodes=3)

    binner.fit(feats, labels)
    np.testing.assert_equal(binner.transform(feats, class_index=1), [0.0, 1.0, 0.5, 0.5, 0.5, 0.5])


def test_fit_transform():
    feats = [0, 1, 2, np.nan, 5, np.nan]
    labels = [0, 1, 0, 1, 1, 0]

    binner = tfb.DecisionTreeBinner('test', max_leaf_nodes=3)
    trans = binner.fit_transform(feats, labels, class_index=1)
    np.testing.assert_equal(trans, [0.0, 1.0, 0.5, 0.5, 0.5, 0.5])
