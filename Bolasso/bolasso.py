from __future__ import division

from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
import numpy as np
import sys
import time


class Bolasso(object):
    """Bolasso feature selection technique, based on the article

    `F. R. Bach, Bolasso: model consistent Lasso estimation through the bootstrap, ICML '08`

    This feature selection wrapper trains a `num_bootstrap` sklearn LogisticRegressionCV classifiers with L1-penalty
    on a bootstrapped subset of the data with size `bootstrap_fraction`. It will store an internal Dataframe with the
    raw coefficients of the trained classifiers and exposes a method to get some summary statistics of the full Bolasso
    DF.

    Parameters
    ----------
    num_boostraps: Number of bootstrapped classifiers that will be trained

    bootstrap_fraction: Fraction of the data that will be bootstrap sampled with replacement.

    random_seed: Fix the seed for the bootstrap generator

    kwargs: All the arguments that the sklearn LogisticRegressionCV classifier takes.

    Attributes
    ----------
    logit: The initialized LogisticRegressionCV classifier

    bolasso_df: The internal DataFrame holding all the individual coefficients

    """

    def __init__(self, num_bootstraps, bootstrap_fraction, random_seed=None, **kwargs):
        self.initialized = False

        self.num_bootstraps = num_bootstraps
        self.bootstrap_fraction = bootstrap_fraction
        self.random_seed = random_seed

        self.Cs = kwargs.get('Cs', 10)
        self.fit_intercept = kwargs.get('fit_intercept', True)
        self.cv = kwargs.get('cv', None)
        self.dual = kwargs.get('dual', False)
        self.scoring = kwargs.get('scoring', None)
        self.tol = kwargs.get('tol', 1e-4)
        self.max_iter = kwargs.get('max_iter', 100)
        self.class_weight = kwargs.get('class_weight', None)
        self.n_jobs = kwargs.get('n_jobs', 1)
        self.verbose = kwargs.get('verbose', 0)
        self.refit = kwargs.get('refit', True)
        self.intercept_scaling = kwargs.get('intercept_scaling', 1.0)
        self.multi_class = kwargs.get('multi_class', 'ovr')
        self.random_state = kwargs.get('random_state', None)

        # The following parameters are changed from default
        # since we want to induce sparsity in the final
        # feature set of Bolasso.
        # liblinear is needed to be working with 'L1' penalty.
        self.logit = LogisticRegressionCV(
            Cs=self.Cs,
            fit_intercept=self.fit_intercept,
            cv=self.cv,
            dual=self.dual,
            penalty='l1',
            scoring=self.scoring,
            solver='liblinear',
            tol=self.tol,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            refit=self.refit,
            intercept_scaling=self.intercept_scaling,
            multi_class=self.multi_class,
            random_state=self.random_state
        )

        self.attributes = None
        self.bolasso_df = None

    def _init_bolasso_df(self, data, labels):
        self.attributes = data.columns
        self.bolasso_df = pd.DataFrame(columns=self. attributes)

        self.initialized = True

        return data.as_matrix(), labels.values

    def _get_bootstrap_sample(self, data, labels):
        if self.random_seed:
            np.random.seed(self.random_seed)
        indices = range(len(labels))
        rand_idx = np.random.choice(indices, size=int(self.bootstrap_fraction * len(labels)), replace=True)

        return data[rand_idx, :], labels[rand_idx]

    def fit(self, data, labels):
        """
        Fits the Bolasso selector

        :param data: Pandas DataFrame containing all the data
        :param labels: Pandas Series with the labels
        :return: None
        """

        if not self.initialized:
            data, labels = self._init_bolasso_df(data, labels)
        else:
            if isinstance(data, pd.core.frame.DataFrame):
                data = data.as_matrix()
            if isinstance(labels, pd.core.series.Series):
                labels = labels.values

        start = time.time()
        for m in range(self.num_bootstraps):

            boot_data, boot_labels = self._get_bootstrap_sample(data, labels)
            self.logit.fit(boot_data, boot_labels)
            self.bolasso_df = self.bolasso_df.append(pd.Series({attr: coef for attr, coef in zip(self.attributes, self.logit.coef_.flatten().tolist())}), ignore_index=True)

            sys.stdout.write("\r>> Iteration: {0:0d}/{1:0d}, elapsed time:{2:4.1f} s".format(m+1, self.num_bootstraps, time.time()-start))
            sys.stdout.flush()

    def _calculate_stats(self, row):
        return pd.Series({'coef_mean': row.mean(),
                          'coef_std': row.std(),
                          'num_occurence': (row != 0).sum(),
                          'frac_occurence': (row != 0).sum() / self.num_bootstraps})

    def get_feature_stats(self):
        """
        Returns summary statistics like mean and number of occurences for the Bolasso selected features

        :return: Pandas DataFrame
        """
        if self.initialized:
            return self.bolasso_df.apply(self._calculate_stats).T.sort_values(by='num_occurence', ascending=False)
        else:
            raise ValueError("You need to fit the selector first.")

