from __future__ import division
import numpy as np


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
