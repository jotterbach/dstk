from __future__ import division
import sklearn
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from DSTK.FeatureBinning._utils import _process_nan_values, _preprocess_values_and_targets

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


def _convert_to_conditional_proba_buckets(sorted_nodes):
    return [(bucket, (vals / np.sum(vals)).tolist()) for bucket, vals in sorted_nodes]


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


def _get_probability_for_value_in_bucket_and_class(cond_proba_buckets, value, class_index=1):
    assert cond_proba_buckets is not None, "Conditional probability bins cannot be None"

    # binning method guarantees that last bucket is nan bucket
    if np.isnan(value):
        if np.isnan(cond_proba_buckets[-1][0]):
            return cond_proba_buckets[-1][1][class_index]
        else:
            raise ValueError("Conditional probability bins do not contain NaN assignment")

    # since we check for values greater than the lower boundary
    # we have to explicitly handle the case of negative infinity
    if np.isneginf(value):
        return cond_proba_buckets[0][1][class_index]

    else:
        for bucket in cond_proba_buckets[:-1]:
            boundaries = bucket[0]
            if (value > boundaries[0]) & (value <= boundaries[1]):
                return bucket[1][class_index]

    raise RuntimeError('Cannot determine bucket for value: {}'.format(value))


def _get_bucket_for_value(cond_proba_buckets, value):
    assert cond_proba_buckets is not None, "Conditional probability bins cannot be None"

    # binning method guarantees that last bucket is nan bucket
    if np.isnan(value):
        if np.isnan(cond_proba_buckets[-1][0]):
            return 'is_nan'
        else:
            raise ValueError("Conditional probability bins do not contain NaN assignment")

    # since we check for values greater than the lower boundary
    # we have to explicitly handle the case of negative infinity
    if np.isneginf(value):
        return str(cond_proba_buckets[0]).replace(')', ']')

    else:
        for bucket in cond_proba_buckets[:-1]:
            boundaries = bucket[0]
            if (value > boundaries[0]) & (value <= boundaries[1]):
                return str(bucket[0]).replace(')', ']')

    raise RuntimeError('Cannot determine bucket for value: {}'.format(value))


class TreeBasedFeatureBinning(object):

    def __init__(self, name, **kwargs):
        self.name = name
        self._count_buckets = None
        self.cond_proba_buckets = None
        self.is_fit = False

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

    def fit(self, values, target):
        """
        This utility helps you to compute feature bins using a classification tree approach. It turns the leaves into non-equidistant
        bin cond_proba_buckets and calculates the frequentist's probability of a feature having a certain label and being in a certain bin.
        Note that it can handle NaN by assigning it its own range.

        E.g.

        >>> feats = [0, 1, 2, np.nan, 5, np.nan]
        >>> labels = [0, 1, 0, 1, 1, 0]

        then

        >>> tfb = TreeBasedFeatureBinning()
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

        non_nan_feats, non_nan_labels = _preprocess_values_and_targets(values, target)

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
                self._count_buckets = sorted(lst_of_bins, key=lambda x: x[0][0])
                self.cond_proba_buckets = _convert_to_conditional_proba_buckets(self._count_buckets)
            else:
                self.cond_proba_buckets = [((np.NINF, np.PINF), label_dist / np.max(label_dist))]

        # handle case of purely nan feature values
        else:
            self.cond_proba_buckets = []
            prior = None

        self.cond_proba_buckets.append((np.nan, _process_nan_values(values, target, prior)))
        self.is_fit = True
        return self

    def transform(self, feature_values, class_index=1):
        """
        See output of 'fit_transform()'
        :param feature_values:
        :param class_index:
        :return:
        """
        assert self.is_fit, "FeatureBinner has to be fit to the data first"

        return [_get_probability_for_value_in_bucket_and_class(self.cond_proba_buckets,
                                                                                       val,
                                                                                       class_index)
                for val in feature_values]

    def fit_transform(self, feature_values, target_values, class_index=1):
        """
        This utility helps you to calculate the conditional probability of a feature with value x in a bucket X to be of class T P(T | x in X).
        The buckets are created using the `tree_based_conditional_probability_bins` utility function.

        E.g.

        >>> feats = [0, 1, 2, np.nan, 5, np.nan]
        >>> labels = [0, 1, 0, 1, 1, 0]

        then

        >>> tfb = TreeBasedFeatureBinning()
        >>> tfb.fit_transform(feats, labels, max_leaf_nodes = 3)
        ... [0.0, 1.0, 0.5, 0.5, 0.5, 0.5]

        which we can interpret as: given an example with label 0 there is a 25% probability of having a feature value in the range
        (-inf, 0.5).

        :param feature_values: list or array of the feature values
        :param target_values: list or array of the corresponding labels
        :param class_index: Index of the corresponding class in the conditional probability vector for each bucket.
               Defaults to 1 (as mostly used for binary classification)
        :param kwargs: supports all keywords of an sklearn.DecisionTreeClassifier
        :return: list of cond_proba_buckets with corresponding conditional probabilities P( T | x in X )
                 for a given example with value x in bin with range X to have label T and list of conditional probabilities for each value to be of class T
        """
        self.fit(feature_values, target_values)
        return self.transform(feature_values, class_index=class_index)

    def transform_to_categorical(self, feature_values):
        """
        Returns the list of buckets the given feature values fall into.
        E.g.

        >>> feats = [0, 1, 2, np.nan, 5, np.nan]
        >>> labels = [0, 1, 0, 1, 1, 0]

        then

        >>> tfb = TreeBasedFeatureBinning()
        >>> tfb.transform_to_categorical(feats, labels)
        ... ['(-inf, 0.5]', '(0.5, 1.5]', '(1.5, 3.5]', 'is_nan', '(3.5, inf]', 'is_nan']


        :param feature_values: list of features
        :return: list of strings that denote the bucket of a feature value
        """

        assert self.is_fit, "FeatureBinner has to be fit to the data first"

        return [_get_bucket_for_value(self.cond_proba_buckets, val)for val in feature_values]


class CategoricalNaiveBayesBinner(object):

    def __init__(self, name):
        self.name = name
        self.categories = list()
        self._count_buckets = list()
        self.cond_proba_buckets = list()
        self.is_fit = False

    def fit(self, categories, targets):

        assert (categories is not None) & (targets != []), "feature_values cannot be None or empty"
        assert (categories is not None) & (targets != []), "target_values cannot be None or empty"
        assert len(categories) == len(targets), "feature_values and target_values must have same length"

        categories = np.asarray(categories, dtype='str')

        self.categories = np.unique(categories).tolist()

        cat_to_idx_dct = {cat: np.where(cat == categories)[0] for cat in self.categories}

        for cat, grp_idx in cat_to_idx_dct.iteritems():
            mean = targets[grp_idx].mean()
            pos_count = targets[grp_idx].sum()
            neg_count = targets[grp_idx].size - pos_count
            self._count_buckets.append((cat, [neg_count, pos_count]))
            self.cond_proba_buckets.append((cat, [1-mean, mean]))

        self._count_buckets.append(('N/A', [targets.size - targets.sum(), targets.sum()]))
        self.cond_proba_buckets.append(('N/A', [1 - targets.mean(), targets.mean()]))
        self.cond_proba_buckets = sorted(self.cond_proba_buckets, key=lambda x: x[0])
        self.is_fit = True
        return self

    def transform(self, feature_values, class_index=1):
        assert self.is_fit, "FeatureBinner has to be fit to the data first"

        return [self._get_probability_for_value_in_category_and_class(val, class_index) for val in feature_values]

    def fit_transform(self, feature_values, target_values, class_index=1):
        self.fit(feature_values, target_values)
        return self.transform(feature_values, class_index=class_index)

    def _get_probability_for_value_in_category_and_class(self, cat, class_index):
        if cat in self.categories:
            for cond_cat, proba in self.cond_proba_buckets:
                if cat == cond_cat:
                    return proba[class_index]
        else:
            for cond_cat, proba in self.cond_proba_buckets:
                if cond_cat == 'N/A':
                    return proba[class_index]
