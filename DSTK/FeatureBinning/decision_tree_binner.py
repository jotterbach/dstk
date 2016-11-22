from __future__ import division
import sklearn
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from DSTK.FeatureBinning._utils import _process_nan_values, _get_non_nan_values_and_targets
from DSTK.FeatureBinning.base_binner import BaseBinner


def _recurse_tree(tree, lst, mdlp, node_id=0, depth=0, min_val=np.NINF, max_val=np.PINF):
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    if left_child == sklearn.tree._tree.TREE_LEAF:
        lst.append(((min_val, max_val), tree.value[node_id].flatten().tolist()))
        return
    else:
        if mdlp and _check_mdlp_stop(tree, node_id):
            lst.append(((min_val, max_val), tree.value[node_id].flatten().tolist()))
            return
        _recurse_tree(tree, lst, mdlp, left_child, depth=depth + 1, min_val=min_val, max_val=tree.threshold[node_id])

    if right_child == sklearn.tree._tree.TREE_LEAF:
        lst.append(((min_val, max_val), tree.value[node_id].flatten().tolist()))
        return
    else:
        if mdlp and _check_mdlp_stop(tree, node_id):
            lst.append(((min_val, max_val), tree.value[node_id].flatten().tolist()))
            return
        _recurse_tree(tree, lst, mdlp, right_child, depth=depth + 1, min_val=tree.threshold[node_id], max_val=max_val)


def _convert_count_buckets_to_split_and_vals(sorted_nodes):
    splits = [bucket[1] for bucket, vals in sorted_nodes]
    values = [(vals / np.sum(vals)).tolist() for bucket, vals in sorted_nodes]
    return splits, values


def _check_mdlp_stop(tree, node_id):
    """
    The MDLP implementation follows the paper of

        U. S. Fayyad and K. B. Irani, Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning, JPL TRS 1992
        http://hdl.handle.net/2014/35171
    """

    num_samples = tree.value[node_id].flatten().sum()

    gain = _calculate_gain(tree, node_id)
    delta = _calculate_noise_delta(tree, node_id)

    return gain < (delta + np.log2(num_samples - 1)) / num_samples


def _calculate_entropy(array):
    non_zero_array = array / array.sum()
    return -1 * np.sum(non_zero_array * np.log2(non_zero_array))


def _calculate_gain(tree, node_id):
    S, nS, S1, nS1, S2, nS2 = _get_variables_for_entropy_calculation(tree, node_id)

    return _calculate_entropy(S) \
            - S1.sum() / S.sum() * _calculate_entropy(S1) \
            - S2.sum() / S.sum() * _calculate_entropy(S2)


def _calculate_noise_delta(tree, node_id):
    S, nS, S1, nS1, S2, nS2 = _get_variables_for_entropy_calculation(tree, node_id)

    return np.log2(np.power(3, nS) - 2) \
            - (nS * _calculate_entropy(S)
            - nS1 * _calculate_entropy(S1)
            - nS2 * _calculate_entropy(S2))


def _get_variables_for_entropy_calculation(tree, node_id):
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    full_set_values = tree.value[node_id].flatten()
    left_set_values = tree.value[left_child].flatten()
    right_set_values = tree.value[right_child].flatten()

    # remove zeros from value_counts to continue processing
    full_set_without_zero_counts = full_set_values[np.where(full_set_values > 0)[0]]
    full_set_tree_classes = full_set_without_zero_counts.size

    left_set_without_zero_counts = left_set_values[np.where(left_set_values > 0)[0]]
    left_set_tree_classes = left_set_without_zero_counts.size

    right_set_without_zero_counts = right_set_values[np.where(right_set_values > 0)[0]]
    right_set_tree_classes = right_set_without_zero_counts.size

    return full_set_without_zero_counts, full_set_tree_classes, left_set_without_zero_counts, left_set_tree_classes, right_set_without_zero_counts, right_set_tree_classes


class DecisionTreeBinner(BaseBinner):

    @property
    def is_fit(self):
        return self._is_fit

    @property
    def values(self):
        return self._values

    @property
    def splits(self):
        return self._splits

    @is_fit.setter
    def is_fit(self, is_fit):
        self._is_fit = is_fit

    @values.setter
    def values(self, values):
        self._values = values

    @splits.setter
    def splits(self, splits):
        self._splits = splits

    def __init__(self, name, **kwargs):
        self.name = name
        self._is_fit = False

        criterion = kwargs.get('criterion', 'gini')
        splitter = kwargs.get('splitter', 'best')
        max_depth = kwargs.get('max_depth', None)
        min_samples_split = kwargs.get('min_samples_split', 2)
        min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        min_weight_fraction_leaf = kwargs.get('min_weight_fraction_leaf', 0.0)
        max_features = kwargs.get('max_features', None)
        random_state = kwargs.get('random_state', None)
        max_leaf_nodes = kwargs.get('max_leaf_nodes', None)
        class_weight = kwargs.get('class_weight', None)
        presort = kwargs.get('presort', False)

        self.mdlp = kwargs.get('mdlp', False)

        if self.mdlp:
            criterion = 'entropy'
            max_leaf_nodes = None
            max_depth = None

        self.dtc = DecisionTreeClassifier(criterion=criterion,
                                          splitter=splitter,
                                          max_depth=max_depth,
                                          min_samples_split=min_samples_split,
                                          min_samples_leaf=min_samples_leaf,
                                          min_weight_fraction_leaf=min_weight_fraction_leaf,
                                          max_features=max_features,
                                          random_state=random_state,
                                          max_leaf_nodes=max_leaf_nodes,
                                          class_weight=class_weight,
                                          presort=presort)

        self._splits = [np.PINF]
        self._values = list()

    def fit(self, values, target):
        """
        This utility helps you to compute feature bins using a classification tree approach. It turns the leaves into non-equidistant
        bin cond_proba_buckets and calculates the frequentist's probability of a feature having a certain label and being in a certain bin.
        Note that it can handle NaN by assigning it its own range.

        E.g.

        >>> feats = [0, 1, 2, np.nan, 5, np.nan]
        >>> labels = [0, 1, 0, 1, 1, 0]

        then

        >>> tfb = DecisionTreeBinner()
        >>> tfb.fit(feats, labels, max_leaf_nodes = 3)
        >>> tfb.cond_proba_buckets
        ... [((-inf, 0.5), [1.0, 0.0]),
        ...  ((0.5, 1.5), [0.0, 1.0]),
        ...  ((1.5, inf), [0.5, 0.5]),
        ...  (nan, [0.5, 0.5])]

        which we can interpret as: given an example with label 0 there is a 25% probability of having a feature value in the range
        (-inf, 0.5).

        :param feature_values: list or array of the feature values
        :param target_values: list or array of the corresponding labels
        :param kwargs: supports all keywords of an sklearn.DecisionTreeClassifier
        :return: Nothing, but sets the self.cond_proba_buckets field
        """

        assert (values is not None) & (values != []), "feature_values cannot be None or empty"
        assert (target is not None) & (target != []), "target_values cannot be None or empty"
        assert len(values) == len(target), "feature_values and target_values must have same length"

        non_nan_feats, non_nan_labels = _get_non_nan_values_and_targets(values, target)

        if non_nan_feats.size > 0:
            self.dtc.fit(non_nan_feats, non_nan_labels)

            prior = np.asarray(non_nan_labels).mean()

            # Handle case of pure classes after nan treatment explicitly
            # To this end we check if we have a binary classification problem, and if so
            # we check that we have counts in both classes so we can actually build a tree that is not a single leaf
            label_dist = np.bincount(non_nan_labels.flatten())
            if (np.min(label_dist) > 0) & (len(label_dist) == 2):
                lst_of_bins = list()
                # recurse edits the list inplace
                _recurse_tree(self.dtc.tree_, lst_of_bins, self.mdlp)
                _count_buckets = sorted(lst_of_bins, key=lambda x: x[0][0])
                self._splits, self._values = _convert_count_buckets_to_split_and_vals(_count_buckets)

        # handle case of purely nan feature values
        else:
            prior = None

        self._splits.append(np.NaN)
        self._values.append(_process_nan_values(values, target, prior))
        self.is_fit = True
        return self
