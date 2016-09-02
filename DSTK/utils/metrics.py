from __future__ import division

import numpy as np
import pandas as pd


def _calculate_tp_tn_fp_fn(actual_label, predicted_label, pos_class_label, neg_class_label):
    tp = len(set(np.where(predicted_label == pos_class_label)[0]).intersection(
        set(np.where(actual_label == pos_class_label)[0])))
    tn = len(set(np.where(predicted_label == neg_class_label)[0]).intersection(
        set(np.where(actual_label == neg_class_label)[0])))
    fp = len(set(np.where(predicted_label == pos_class_label)[0]).intersection(
        set(np.where(actual_label == neg_class_label)[0])))
    fn = len(set(np.where(predicted_label == neg_class_label)[0]).intersection(
        set(np.where(actual_label == pos_class_label)[0])))

    return tp, tn, fp, fn


def _calculate_metrics(tp, tn, fp, fn):
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall \
            = 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    support = int(tp + fn)

    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    prior = (tp + fn) / (tp + tn + fp + fn)

    prevalence = (tp + fp) / (tp + tn + fp + fn)

    return prevalence, precision, recall, accuracy, f1, prior, support


def get_all_metrics(actual_label, predicted_label, **kwargs):

    pos_class_label = kwargs.get('pos_class_label', 1)
    pos_class_name = kwargs.get('pos_class_name', str(pos_class_label))

    neg_class_label = kwargs.get('neg_class_label', 0)
    neg_class_name = kwargs.get('neg_class_name', str(neg_class_label))

    assert pos_class_label != neg_class_label, "class labels cannot be the same"
    assert pos_class_name != neg_class_name, "class names cannot be the same"

    predicted_label = np.asarray(predicted_label, dtype=int)
    actual_label = np.asarray(actual_label, dtype=int)

    return _calculate_metrics(*_calculate_tp_tn_fp_fn(actual_label, predicted_label, pos_class_label, neg_class_label))


def confusion_matrix(actual_label, predicted_label, **kwargs):

    pos_class_label = kwargs.get('pos_class_label', 1)
    pos_class_name = kwargs.get('pos_class_name', str(pos_class_label))

    neg_class_label = kwargs.get('neg_class_label', 0)
    neg_class_name = kwargs.get('neg_class_name', str(neg_class_label))

    assert pos_class_label != neg_class_label, "class labels cannot be the same"
    assert pos_class_name != neg_class_name, "class names cannot be the same"

    predicted_label = np.asarray(predicted_label, dtype=int)
    actual_label = np.asarray(actual_label, dtype=int)

    tp, tn, fp, fn = _calculate_tp_tn_fp_fn(actual_label, predicted_label, pos_class_label, neg_class_label)

    idx = pd.MultiIndex.from_tuples([('actual', pos_class_name), ('actual', neg_class_name)])
    cols = pd.MultiIndex.from_tuples([('predicted', pos_class_name), ('predicted', neg_class_name)])

    return pd.DataFrame(np.array([[tp, fn], [fp, tn]]), index=idx, columns=cols)


def classification_report(actual_label, predicted_label, **kwargs):

    pos_class_label = kwargs.get('pos_class_label', 1)
    pos_class_name = kwargs.get('pos_class_name', str(pos_class_label))

    neg_class_label = kwargs.get('neg_class_label', 0)
    neg_class_name = kwargs.get('neg_class_name', str(neg_class_label))

    assert pos_class_label != neg_class_label, "class labels cannot be the same"
    assert pos_class_name != neg_class_name, "class names cannot be the same"

    predicted_label = np.asarray(predicted_label, dtype=int)
    actual_label = np.asarray(actual_label, dtype=int)

    idx = [pos_class_name, neg_class_name]
    cols = ['prevalence', 'precision', 'recall', 'accuracy', 'f1', 'prior', 'support']

    arr = np.array([[item for item in _calculate_metrics(*_calculate_tp_tn_fp_fn(actual_label, predicted_label, pos_class_label, neg_class_label))],
                    [item for item in _calculate_metrics(*_calculate_tp_tn_fp_fn(actual_label, predicted_label, neg_class_label, pos_class_label))]])

    df = pd.DataFrame(arr, index=idx, columns=cols)
    df.index.name = 'pos_class'

    return df


def prevalence_precision_curve(actual_label, predicted_proba, **kwargs):
    pos_class_label = kwargs.get('pos_class_label', 1)
    pos_class_name = kwargs.get('pos_class_name', str(pos_class_label))

    neg_class_label = kwargs.get('neg_class_label', 0)
    neg_class_name = kwargs.get('neg_class_name', str(neg_class_label))

    assert pos_class_label != neg_class_label, "class labels cannot be the same"
    assert pos_class_name != neg_class_name, "class names cannot be the same"

    prev_prec_tup = [_calculate_metrics(*_calculate_tp_tn_fp_fn(actual_label,  np.asarray(np.asarray(predicted_proba) >= t, dtype=int), pos_class_label, neg_class_label))[:2] for t in np.arange(0, 1, 0.01)]
    prevalence = [tup[0] for tup in prev_prec_tup]
    precision = [tup[1] for tup in prev_prec_tup]
    return prevalence, precision
