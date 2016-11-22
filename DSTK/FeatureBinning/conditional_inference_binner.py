from __future__ import division
import numpy as np
import scipy.stats as st
from DSTK.FeatureBinning.base_binner import BaseBinner
from DSTK.FeatureBinning._utils import _naive_bayes_bins, _filter_special_values


class _Node(object):

    def __init__(self, id, threshold, left_child, right_child, is_leaf, counts, statistic):
        self.id = id
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = is_leaf
        self.counts = counts
        self.statistic = statistic
        self.cond_proba = counts / counts.sum()
        self.log_odds = np.log(counts[1] / counts[0])

    def __str__(self):
        return "id: {}, threshold: {}, left_id: {}, right_id: {}, leaf: {}, counts: {}, statistic: {}, probas: {}, log_odds: {}".format(self.id, self.threshold, self.left_child, self.right_child, self.is_leaf, self.counts, self.statistic, self.cond_proba, self.log_odds)


class ConditionalInferenceBinner(BaseBinner):
    """
    This Binner is inspired by the `partykit::ctree` package of R. It used conditional inference for recursive partitioning of an input space.
    The method is based on:
        T. Horton, K. Hornik, A. Zeileis, "Unbiased Recursive Partitioning: A Conditional Inference Framework"
        Journal of Computational and Graphical Statistics 15, 651-674 (2006)
        http://www.tandfonline.com/doi/abs/10.1198/106186006X133933

    The basic idea is to leverage PearsonR correlation coefficient to find the best split (by maximizing the absolute value of the correlation
    and terminate the regression based on accepting the Null-Hypothesis of the PearsonR coefficient under a permutation test.
    """
    @property
    def values(self):
        return self._values

    @property
    def splits(self):
        return self._splits

    @property
    def is_fit(self):
        return self._is_fit

    @values.setter
    def values(self, values):
        self._values = values

    @splits.setter
    def splits(self, splits):
        self._splits = splits

    @is_fit.setter
    def is_fit(self, is_fit):
        self._is_fit = is_fit

    def __init__(self, name, **kwargs):
        self.name = name

        self.alpha = kwargs.get('alpha', 0.05)
        self.min_samples_split = kwargs.get('min_samples_split', 2)
        self.min_samples_leaf = kwargs.get('min_samples_leaf', 2)
        self.special_values = kwargs.get('special_values', [np.NaN])

        self.num_classes = None

        self._splits = [np.PINF]
        self._values = list()
        self.nodes = list()

        self._is_fit = False

    def fit(self, values, targets):
        """
        Fits the binner. It turns the leaves into non-equidistant bin cond_proba_buckets and calculates the frequentist's probability of a feature having a certain label and being in a certain bin. Note that it can handle NaN by assigning it its own range.

        :param feature_values: list or array of the feature values (potentially containing NaN)
        :param target_values: list or array of the corresponding labels
        :param kwargs: supports all keywords of an sklearn.DecisionTreeClassifier
        :return: self with fitted bins.
        """

        assert (values is not None) & (values != []), "feature_values cannot be None or empty"
        assert (targets is not None) & (targets != []), "target_values cannot be None or empty"
        assert len(values) == len(targets), "feature_values and target_values must have same length"

        values = values.astype(np.float32)
        targets = targets.astype(np.int32)

        self.num_classes = len(np.bincount(targets))

        if self.num_classes == 1:
            raise ArithmeticError("data contains only one label.")

        special_vals_idx = _filter_special_values(values, self.special_values)
        non_special_vals, non_special_labels = values[special_vals_idx['regular']], targets[special_vals_idx['regular']]

        if non_special_vals.size > 0:
            self._recurse(non_special_vals, non_special_labels, 0)
            self._calculate_conditional_probas(non_special_vals, non_special_labels)
            prior = np.bincount(non_special_labels)[1] / len(non_special_labels)
        else:
            prior = None

        for val in self.special_values:
            self.add_bin(val, _naive_bayes_bins(targets[special_vals_idx[str(val)]], prior))

        self.is_fit = True
        return self

    def _calculate_conditional_probas(self, values, targets):

        idx = np.digitize(values, self._splits, right=True)
        for bin_idx, bin in enumerate(np.unique(idx)):
            counts = np.bincount(targets[np.where(idx == bin)[0]], minlength=self.num_classes)
            self._values.append(list(counts / counts.sum()))

    def _recurse(self, values, targets, node_id):
        new_vals = values.copy()
        new_targets = targets.copy()

        counts = np.bincount(new_targets, minlength=self.num_classes)
        statistic = ConditionalInferenceBinner._get_statistic(new_vals, new_targets)

        if self._terminate_recursion(new_vals, new_targets):
            self._create_leaf(node_id, counts, statistic)

        else:
            split = self._find_split(new_vals, new_targets)
            if split:
                self._append_split(split)

                left_idx = np.where(new_vals <= split)[0]
                right_idx = np.where(new_vals > split)[0]

                self._recurse(new_vals[left_idx], new_targets[left_idx], node_id=node_id+1)
                self._recurse(new_vals[right_idx], new_targets[right_idx], node_id=node_id+2)

                self.nodes.append(_Node(node_id, split, node_id + 1, node_id + 2, False, counts, statistic))
            else:
                self._create_leaf(node_id, counts, statistic)

    def _create_leaf(self, node_id, counts, statistic):
        self.nodes.append(_Node(node_id, np.NaN, -1, -1, True, counts, statistic))

    def _append_split(self, split_value):
        if self._splits == list():
            self._splits.append(split_value)
        else:
            idx = np.digitize(split_value, self._splits)
            self._splits = np.insert(self._splits, idx, split_value).astype(np.float32).tolist()

    def _terminate_recursion(self, values, targets):
        accept_null_hypothesis = not self._reject_null_hypothesis(values, targets, self.alpha)
        is_not_splittable = not ConditionalInferenceBinner._splittable(values, targets, self.min_samples_split)
        return accept_null_hypothesis or is_not_splittable

    @staticmethod
    def _splittable(values, targets, min_samples):
        return (len(np.unique(targets) >= 2) and (len(np.unique(values)) > min_samples))

    def _find_split(self, values, targets):
        unique_vals = np.unique(values).astype(np.float64)
        split_candidates = unique_vals[:-1]
        lst_of_splits = list()
        for cand in split_candidates:
            sub_set = (values <= cand).astype(np.int32)
            if self._check_acceptable_split(values, targets, np.where(sub_set)[0]):
                lst_of_splits.append((cand, np.abs(ConditionalInferenceBinner._get_statistic(sub_set, targets)).astype(np.float64)))

        lst_of_splits = sorted(lst_of_splits, key=lambda tup: tup[1], reverse=True)
        if lst_of_splits:
            return lst_of_splits[0][0]
        else:
            return None

    def _check_acceptable_split(self, values, targets, index):
        return ((len(index) >= self.min_samples_leaf) and
                (len(np.unique(targets[index])) > 1) and
                (len(np.unique(values[index])) > 1) and
                (len(values) - len(index) >= self.min_samples_leaf))

    @staticmethod
    def _reject_null_hypothesis(values, target, alpha):
        return np.abs(ConditionalInferenceBinner._get_pvalue(values, target)) <= alpha

    @staticmethod
    def _get_statistic(values, target):
        return st.pearsonr(values, target)[0]

    @staticmethod
    def _get_pvalue(values, target):
        return st.pearsonr(values, target)[1]
