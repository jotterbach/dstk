from __future__ import division

from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from base_selector import BaseSelector
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
import numpy as np


class Bolasso(BaseSelector):
    """Bolasso feature selection technique, based on the article

    `F. R. Bach, Bolasso: model consistent Lasso estimation through the bootstrap, ICML '08`

    This feature selection wrapper trains a `num_bootstrap` sklearn LogisticRegressionCV classifiers with L1-penalty
    on a bootstrapped subset of the data with size `bootstrap_fraction`. It will store an internal Dataframe with the
    raw coefficients of the trained classifiers and exposes a method to get some summary statistics of the full Bolasso
    DF.

    Parameters
    ----------
    bootstrap_fraction: Fraction of the data that will be bootstrap sampled with replacement.

    random_seed: Fix the seed for the bootstrap generator

    kwargs: All the arguments that the sklearn LogisticRegressionCV classifier takes.

    Attributes
    ----------
    logit: The initialized LogisticRegressionCV classifier

    coeff_df: The internal DataFrame holding all the individual coefficients

    """

    def __init__(self, bootstrap_fraction, random_seed=None, feature_importance_metric=None, feature_importance_threshold=None, **kwargs):

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

        super(Bolasso, self).__init__(bootstrap_fraction, self.logit, random_seed=random_seed, feature_importance_metric=feature_importance_metric, feature_importance_threshold=feature_importance_threshold)

    def _get_feature_coeff(self):
        return self.logit.coef_.flatten().tolist()


class SGDBolasso(BaseSelector, BaseEstimator):
    """Bolasso feature selection technique, based on the article

    `F. R. Bach, Bolasso: model consistent Lasso estimation through the bootstrap, ICML '08`

    This feature selection wrapper trains a `num_bootstrap` sklearn SGDClassifier classifiers with L1-penalty
    on a bootstrapped subset of the data with size `bootstrap_fraction`. It will store an internal Dataframe with the
    raw coefficients of the trained classifiers and exposes a method to get some summary statistics of the full Bolasso
    DF.

    Parameters
    ----------
    bootstrap_fraction: Fraction of the data that will be bootstrap sampled with replacement.

    random_seed: Fix the seed for the bootstrap generator

    kwargs: All the arguments that the sklearn SGDClassifier takes.

    Attributes
    ----------
    sgd_logit: The initialized SGDClassifier classifier

    coeff_df: The internal DataFrame holding all the individual coefficients

    """

    def __init__(self,
                 bootstrap_fraction,
                 random_seed=None,
                 feature_importance_metric=None,
                 feature_importance_threshold=None,
                 alpha=0.0001,
                 fit_intercept=True,
                 n_iter=5,
                 shuffle=True,
                 verbose=0,
                 epsilon=0.1,
                 n_jobs=1,
                 random_state=None,
                 learning_rate="optimal",
                 eta0=0.0,
                 power_t=0.5,
                 class_weight=None,
                 warm_start=False,
                 average=False):

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.verbose = verbose
        self.epsilon = epsilon
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.class_weight = class_weight
        self.warm_start = warm_start
        self.average = average

        # The following parameters are changed from default
        # since we want to induce sparsity in the final
        # feature set of Bolasso.
        self.sgd_logit = SGDClassifier(
            loss="log",
            penalty='l1',
            alpha=self.alpha,
            l1_ratio=1.0,
            fit_intercept=self.fit_intercept,
            n_iter=self.n_iter,
            shuffle=self.shuffle,
            verbose=self.verbose,
            epsilon=self.epsilon,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            power_t=self.power_t,
            class_weight=self.class_weight,
            warm_start=self.warm_start,
            average=self.average
        )

        super(SGDBolasso, self).__init__(bootstrap_fraction, self.sgd_logit, random_seed=random_seed, feature_importance_metric=feature_importance_metric, feature_importance_threshold=feature_importance_threshold)

    def _get_feature_coeff(self):
        return self.sgd_logit.coef_.flatten().tolist()

    def predict_proba(self, data):
        return self.sgd_logit.predict_proba(data)

    def predict(self, data):
        return np.round(self.sgd_logit.predict_proba(data)[:, 1])

    def fit_cv(self, data, labels, cv_params, epochs=10, **kwargs):

        n_jobs = kwargs.get('n_jobs', 1)
        iid = kwargs.get('iid', True)
        refit = kwargs.get('refit', True)
        cv = kwargs.get('cv', None)
        verbose = kwargs.get('verbose', 0)
        pre_dispatch = kwargs.get('pre_dispatch', '2*n_jobs')
        error_score = kwargs.get('error_score', 'raise')
        return_train_score = kwargs.get('return_train_score', True)

        param_dct = self.get_params()
        param_dct.update({'bootstrap_fraction': 1.0})

        rscv = GridSearchCV(SGDBolasso(**param_dct),
                            scoring=make_scorer(accuracy_score),
                            verbose=verbose,
                            param_grid=cv_params,
                            fit_params={'epochs': 1, 'verbose': 0},
                            cv=cv,
                            return_train_score=return_train_score,
                            n_jobs=n_jobs,
                            iid=iid,
                            refit=refit,
                            pre_dispatch=pre_dispatch,
                            error_score=error_score)

        rscv.fit(data, labels)

        param_dct = rscv.best_params_.copy()
        param_dct.update({'bootstrap_fraction': self.bootstrap_fraction})
        best_estim = SGDBolasso(**param_dct)

        best_estim.fit(data, labels, epochs=epochs)
        return best_estim, rscv


class Botree(BaseSelector):
    """Botree feature selection technique, based on the article

       `F. R. Bach, Bolasso: model consistent Lasso estimation through the bootstrap, ICML '08`

       This feature selection wrapper trains a `num_bootstrap` sklearn DecisionTree classifiers
       on a bootstrapped subset of the data with size `bootstrap_fraction`. It will store an internal Dataframe with the
       raw coefficients of the trained classifiers and exposes a method to get some summary statistics of the full Botree
       DF.

       Parameters
       ----------
       bootstrap_fraction: Fraction of the data that will be bootstrap sampled with replacement.

       random_seed: Fix the seed for the bootstrap generator

       kwargs: All the arguments that the sklearn DecisionTreeClassifier takes.

       Attributes
       ----------
       dtc: The initialized DecisionTree classifier

       coeff: The internal DataFrame holding all the individual coefficients

       """

    def __init__(self, bootstrap_fraction, random_seed=None, feature_importance_metric=None, feature_importance_threshold=None, **kwargs):

        self.criterion = kwargs.get('criterion', "gini")
        self.splitter = kwargs.get('splitter', "best")
        self.max_depth = kwargs.get('max_depth', None)
        self.min_samples_split = kwargs.get('min_samples_split', 2)
        self.min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        self.min_weight_fraction_leaf = kwargs.get('min_weight_fraction_leaf', 0.)
        self.max_features = kwargs.get('max_features', None)
        self.max_leaf_nodes = kwargs.get('random_state', None)
        self.class_weight = kwargs.get('max_leaf_nodes', None)
        self.random_state = kwargs.get('class_weight', None)
        self.presort = kwargs.get('presort', False)

        self.dtc = DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            class_weight=self.class_weight,
            random_state=self.random_state,
            presort=self.presort
        )

        super(Botree, self).__init__(bootstrap_fraction, self.dtc, random_seed=random_seed, feature_importance_metric=feature_importance_metric, feature_importance_threshold=feature_importance_threshold)

    def _get_feature_coeff(self):
        return self.dtc.feature_importances_.flatten().tolist()


class Boforest(BaseSelector):
    """Boforest feature selection technique, based on the article

       `F. R. Bach, Bolasso: model consistent Lasso estimation through the bootstrap, ICML '08`

       This feature selection wrapper trains `num_bootstrap` sklearn RandomForest classifiers
       on a bootstrapped subset of the data with size `bootstrap_fraction`. It will store an internal Dataframe with the
       raw coefficients of the trained classifiers and exposes a method to get some summary statistics of the full Botree
       DF.

       Parameters
       ----------
       bootstrap_fraction: Fraction of the data that will be bootstrap sampled with replacement.

       random_seed: Fix the seed for the bootstrap generator

       kwargs: All the arguments that the sklearn RandomForest classifier takes.

       Attributes
       ----------
       rfc: The initialized DecisionTree classifier

       coeff_df: The internal DataFrame holding all the individual coefficients

       """

    def __init__(self, bootstrap_fraction, random_seed=None, feature_importance_metric=None, feature_importance_threshold=None, **kwargs):
        self.n_estimators = kwargs.get('n_estimators', 10)
        self.criterion = kwargs.get('criterion', "gini")
        self.max_depth = kwargs.get('max_depth', None)
        self.min_samples_split = kwargs.get('min_samples_split', 2)
        self.min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        self.min_weight_fraction_leaf = kwargs.get('min_weight_fraction_leaf', 0.)
        self.max_features = kwargs.get('max_features', "auto")
        self.max_leaf_nodes = kwargs.get('max_leaf_nodes', None)
        self.bootstrap = kwargs.get('bootstrap', True)
        self.oob_score = kwargs.get('oob_score', True)
        self.n_jobs = kwargs.get('n_jobs', 1)
        self.random_state = kwargs.get('random_state', None)
        self.verbose = kwargs.get('verbose', 0)
        self.warm_start = kwargs.get('warm_start', False)
        self.class_weight = kwargs.get('class_weight', None)

        self.rfc = RandomForestClassifier(
            self.n_estimators,
            self.criterion,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            self.min_weight_fraction_leaf,
            self.max_features,
            self.max_leaf_nodes,
            self.bootstrap,
            self.oob_score,
            self.n_jobs,
            self.random_state,
            self.verbose,
            self.warm_start,
            self.class_weight
        )

        super(Boforest, self).__init__(bootstrap_fraction, self.rfc, random_seed=random_seed, feature_importance_metric=feature_importance_metric, feature_importance_threshold=feature_importance_threshold)

    def _get_feature_coeff(self):
        return self.rfc.feature_importances_.flatten().tolist()