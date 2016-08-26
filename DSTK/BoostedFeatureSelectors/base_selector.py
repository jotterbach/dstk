from __future__ import division

import pandas as pd
import numpy as np
import sys
import time

import abc


class BaseSelector(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _get_feature_coeff(self):
        raise NotImplementedError

    def __init__(self, bootstrap_fraction, classifier, random_seed=None):
        self.initialized = False

        self.num_bootstraps = 0
        self.bootstrap_fraction = bootstrap_fraction
        self.random_seed = random_seed

        self.clf = classifier

        self.attributes = None
        self.coeff_df = None

    def _init_botree_df(self, data, labels):
        self.attributes = data.columns
        self.coeff_df = pd.DataFrame(columns=self. attributes)

        self.initialized = True

        return data.as_matrix(), labels.values

    def _get_bootstrap_sample(self, data, labels):
        if self.random_seed:
            np.random.seed(self.random_seed)
        indices = range(len(labels))
        rand_idx = np.random.choice(indices, size=int(self.bootstrap_fraction * len(labels)), replace=True)

        return data[rand_idx, :], labels[rand_idx]

    def fit(self, data, labels, epochs=10):
        """
        Fits the boosted selector

        :param data: Pandas DataFrame containing all the data
        :param labels: Pandas Series with the labels
        :param epochs: Number of fitting iterations. Defaults to 10.
        :return: None
        """

        if not self.initialized:
            data, labels = self._init_botree_df(data, labels)
        else:
            if isinstance(data, pd.core.frame.DataFrame):
                data = data.as_matrix()
            if isinstance(labels, pd.core.series.Series):
                labels = labels.values

        start = time.time()
        for m in range(epochs):

            boot_data, boot_labels = self._get_bootstrap_sample(data, labels)
            self.clf.fit(boot_data, boot_labels)
            self.coeff_df = self.coeff_df.append(pd.Series({attr: coef for attr, coef in zip(self.attributes, self._get_feature_coeff())}), ignore_index=True)
            self.num_bootstraps += 1

            sys.stdout.write("\r>> Iteration: {0:0d}/{1:0d}, elapsed time:{2:4.1f} s".format(m+1, epochs, time.time()-start))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()

    def _calculate_stats(self, row):
        return pd.Series({'coef_mean': row[row != 0].mean(),
                          'coef_std': row[row != 0].std(),
                          'num_occurence': (row != 0).sum(),
                          'frac_occurence': (row != 0).sum() / self.num_bootstraps})

    def get_feature_stats(self):
        """
        Returns summary statistics like mean and number of occurences for the boosted-selected features

        :return: Pandas DataFrame
        """
        if self.initialized:
            return self.coeff_df.apply(self._calculate_stats).T.sort_values(by='num_occurence', ascending=False)
        else:
            raise ValueError("You need to fit the selector first.")

