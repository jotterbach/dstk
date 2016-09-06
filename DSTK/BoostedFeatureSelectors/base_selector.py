from __future__ import division

import pandas as pd
import numpy as np
import sys
import time
from DSTK.utils import sampling_helpers as sh
from sklearn.metrics import precision_score, accuracy_score, recall_score, roc_auc_score

import abc


class BaseSelector(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _get_feature_coeff(self):
        raise NotImplementedError

    def __init__(self, bootstrap_fraction, classifier, random_seed=None, feature_importance_metric=None, feature_importance_threshold=None):
        self.initialized = False

        self.num_bootstraps = 0
        self.bootstrap_fraction = bootstrap_fraction
        self.random_seed = random_seed

        self.clf = classifier
        self.feature_importance_metric = feature_importance_metric
        self.feature_importance_threshold = feature_importance_threshold

        if feature_importance_metric:
            assert feature_importance_threshold, "If feature permutation importance is to be calculated, a threshold has to be set!"

        self.attributes = None
        self.coeff_df = None

    def _get_feature_importance_metric_func(self):
        if self.feature_importance_metric == 'accuracy':
            return accuracy_score
        elif self.feature_importance_metric == 'recall':
            return recall_score
        elif self.feature_importance_metric == 'precision':
            return precision_score
        elif self.feature_importance_metric == 'roc_auc':
            return roc_auc_score
        else:
            raise NotImplementedError("The requested metric is not implemented. Allowed are: 'accuracy', 'recall', 'precision', 'roc_auc'.")

    def _init_boselector_df(self, data, labels):
        self.attributes = data.columns
        self.coeff_df = pd.DataFrame()

        self.initialized = True

        return data.as_matrix(), labels.values

    def fit(self, data, labels, epochs=10):
        """
        Fits the boosted selector

        :param data: Pandas DataFrame containing all the data
        :param labels: Pandas Series with the labels
        :param epochs: Number of fitting iterations. Defaults to 10.
        :return: None
        """

        if not self.initialized:
            data, labels = self._init_boselector_df(data, labels)
        else:
            if isinstance(data, pd.core.frame.DataFrame):
                data = data.as_matrix()
            if isinstance(labels, pd.core.series.Series):
                labels = labels.values

        start = time.time()
        for m in range(epochs):

            boot_data, oob_data, boot_labels, oob_labels = sh.random_sample(data,
                                                                            labels,
                                                                            sample_fraction=int(self.bootstrap_fraction * len(labels)),
                                                                            random_seed=self.random_seed)

            self.clf.fit(boot_data, boot_labels)
            boselector_metrics_dct = {attr + '__classifier_feature_coeff': coef for attr, coef in zip(self.attributes, self._get_feature_coeff())}

            if self.feature_importance_metric:
                boselector_metrics_dct = self._calculate_feature_importance(oob_data, oob_labels)

            self.coeff_df = self.coeff_df.append(pd.Series(boselector_metrics_dct), ignore_index=True)
            self.num_bootstraps += 1

            sys.stdout.write("\r>> Iteration: {0:0d}/{1:0d}, elapsed time:{2:4.1f} s".format(m+1, epochs, time.time()-start))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()

    def _calculate_feature_importance(self, oob_data, oob_labels):

        dct = dict()
        for col_idx in np.where(np.asarray(self._get_feature_coeff()) != 0)[0]:
            mean_oob_perm = np.mean(np.asarray([self._get_metrics(sh.permute_column_of_numpy_array(oob_data, col_idx), oob_labels) for i in range(10)]), axis=0)
            oob_value = self._get_metrics(oob_data, oob_labels)
            diff = (oob_value - mean_oob_perm) / oob_value

            dct.update({self.attributes.tolist()[col_idx]: diff})

        return dct

    def _get_metrics(self, oob_data, oob_labels):
        oob_proba = self.clf.predict_proba(oob_data)[:, 1]
        if self.feature_importance_metric == 'roc_auc':
            return self._get_feature_importance_metric_func()(oob_labels, oob_proba)
        else:
            oob_predictions = oob_proba >= self.feature_importance_threshold
            return self._get_feature_importance_metric_func()(oob_labels, oob_predictions)

    def _calculate_stats(self, row):

        metric = 'coef'
        if self.feature_importance_metric:
            metric = self.feature_importance_metric + '_diff'

        return pd.Series({metric + '_mean': row[row != 0].mean(),
                          metric + '_std': row[row != 0].std(),
                          'num_occurence': (row != 0).sum(),
                          'frac_occurence': (row != 0).sum() / self.num_bootstraps})

    def get_feature_stats(self, sort_key = 'frac_occurence'):
        """
        Returns summary statistics like mean and number of occurences for the boosted-selected features

        :return: Pandas DataFrame
        """
        if self.initialized:
            return self.coeff_df.apply(self._calculate_stats).T.sort_values(by=sort_key, ascending=False)
        else:
            raise ValueError("You need to fit the selector first.")

