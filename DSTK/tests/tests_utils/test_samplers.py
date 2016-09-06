from sklearn import datasets as ds
from collections import Counter

from DSTK.utils import sampling_helpers as sh
import numpy as np

cancer_ds = ds.load_breast_cancer()
data = cancer_ds['data'][:, :20]
labels = 2 * cancer_ds['target'] - 1

random_seed = 42


def test_downsample():
    initial_cntr = Counter(labels)
    min_size = np.min(initial_cntr.values())
    down_data, down_labels = sh.downsample_majority_class(data, labels, random_seed=random_seed)

    for down_idx, vec in enumerate(down_data):
        idx = np.where((data == vec).all(axis=1))[0].squeeze()
        assert labels[idx] == down_labels[down_idx]

    cntr = Counter(down_labels)
    assert cntr.get(1) == min_size
    assert cntr.get(1) == cntr.get(-1)


def test_upsample():
    initial_cntr = Counter(labels)
    max_size = np.max(initial_cntr.values())
    up_data, up_labels = sh.upsample_minority_class(data, labels, random_seed=random_seed)

    for down_idx, vec in enumerate(up_data):
        idx = np.where((data == vec).all(axis=1))[0].squeeze()
        assert labels[idx] == up_labels[down_idx]

    cntr = Counter(up_labels)
    assert cntr.get(1) == max_size
    assert cntr.get(1) == cntr.get(-1)


def test_permuter():
    arr = np.array([[1, 1],
                    [2, 2],
                    [3, 3],
                    [4, 4],
                    [5, 5]])

    assert_arr = np.array([[1, 2],
                           [2, 5],
                           [3, 3],
                           [4, 1],
                           [5, 4]])

    assert sh.permute_column_of_numpy_array(arr, 1, random_seed=42).tolist() == assert_arr.tolist()
