from __future__ import division
import numpy as np


def _preprocess_values_and_targets(values, target):
    feats = np.array(values)
    labels = np.array(target)

    return feats[~np.isnan(feats)].reshape(-1, 1), labels[~np.isnan(feats)].reshape(-1, 1)


def _process_nan_values(values, target, prior):

    feats = np.array(values)
    labels = np.array(target)
    num_classes = np.bincount(target).size

    num_nan = len(feats[np.isnan(feats)])
    val_count = np.bincount(labels[np.isnan(feats)], minlength=num_classes)
    if (num_nan == 0) and prior:
        return [1 - prior, prior]
    elif (num_nan == 0) and (prior is None):
        return (np.ones((num_classes,), dtype='float64') / num_classes).tolist()
    return list(val_count / num_nan)
