from __future__ import division
import numpy as np


def _get_non_nan_values_and_targets(values, target):
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


def _filter_special_values(values, filter_values):
    feats = np.array(values)
    filters = np.array(filter_values)

    special_vals_idx_dct = dict()
    for val in filters:
        if np.isnan(val):
            idx = np.where(np.isnan(feats))[0]
        else:
            idx = np.where(feats == val)[0]

        special_vals_idx_dct.update({str(val): idx})

    all_special_idx = list()
    for vals in special_vals_idx_dct.values():
        all_special_idx += vals.tolist()

    all_special_idx = np.asarray(all_special_idx)
    all_idx = np.arange(0, len(feats), 1, dtype=int)
    all_non_special_idx = [idx for idx in all_idx if idx not in all_special_idx]
    special_vals_idx_dct.update({'regular': all_non_special_idx})

    return special_vals_idx_dct


def _naive_bayes_bins(target, prior, num_classes=2):
    if len(target) > 1:
        cnts = np.bincount(target, minlength=num_classes)
        return list(cnts / cnts.sum())
    else:
        return [1 - prior, prior]
