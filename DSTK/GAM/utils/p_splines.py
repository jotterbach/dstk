from __future__ import division
import scipy as sp
import numpy as np
from statsmodels import api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import sys


def _get_percentiles(values, num_percentiles=10):
    dx = 100 / num_percentiles
    percentiles = np.arange(dx, 100 + dx, dx)
    return np.percentile(values, q=percentiles)


def _R(x, z):
    return ((z - 0.5)**2 - 1 / 12) * ((x - 0.5)**2 - 1 / 12) / 4 - ((np.abs(x - z) - 0.5)**4 - 0.5 * (np.abs(x - z) - 0.5)**2 + 7 / 240) / 24

R = np.frompyfunc(_R, 2, 1)


def _sigmoid(x):
    """
    Numerically-stable sigmoid function.
    Taken from:
      - http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
      - http://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
    """
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        return z / (1 + z)

sigmoid = np.frompyfunc(_sigmoid, 1, 1)


def _get_basis_vector(vals, knots, with_intercept=True):
    if with_intercept:
        if isinstance(vals, float):
            return np.asarray([1, vals] + R(vals, knots).tolist())
        else:
            return np.asarray([[1, val] + R(val, knots).tolist() for val in vals])
    else:
        if isinstance(vals, float):
            return np.asarray([vals] + R(vals, knots).tolist())
        else:
            return np.asarray([[val] + R(val, knots).tolist() for val in vals])


class PSpline(object):

    def __init__(self, num_percentiles=10):
        self.num_percentiles = num_percentiles
        self._knots = None
        self.spline = None
        self.basis_matrix = None

    def fit(self, values, targets, penalty=0.0):
        self._knots = _get_percentiles(values, num_percentiles=self.num_percentiles)
        self.basis_matrix = _get_basis_vector(values, self._knots)

        X = np.vstack((self.basis_matrix, np.sqrt(penalty) * self._penalty_matrix()))
        y = np.asarray(targets + np.zeros((self.num_percentiles + 2, 1)).flatten().tolist())

        self.spline = sm.OLS(y, X).fit()
        return self

    # def _get_basis_vector(self, vals):
    #     if isinstance(vals, float):
    #         return np.asarray([1, vals] + R(vals, self._knots).tolist())
    #     else:
    #         return np.asarray([[1, val] + R(val, self._knots).tolist() for val in vals])

    def _penalty_matrix(self):
        S = np.zeros((self.num_percentiles + 2, self.num_percentiles + 2))
        S[2:, 2:] = np.real_if_close(sp.linalg.sqrtm(R.outer(self._knots, self._knots).astype(np.float64)), tol=10**8)
        return S

    def gcv_score(self):
        X = self.spline.model.exog[:-(self.num_percentiles + 2), :]
        n = X.shape[0]
        y = self.spline.model.endog[:n]
        y_hat = self.spline.predict(X)

        hat_matrix_trace = self.spline.get_influence().hat_matrix_diag[:n].sum()

        return n * np.power(y - y_hat, 2).sum() / np.power(n - hat_matrix_trace, 2)

    def predict(self, values):
        return self.spline.predict(self._get_basis_vector(values))


class ClassificationPSplines(object):
    """
    This class implements learning a cubic spline interpolation for a binary classification problem. The implementation
    is based on the P-IRLS algorithm as described in the book:

        Simon N. Woods, Generalized Additive Models, Chapman and Hall/CRC (2006)
    """

    def __init__(self, **kwargs):
        self.num_percentiles = kwargs.get('num_percentiles', 10)
        self.tol = kwargs.get('tol', 5e-4)
        self.max_iter = kwargs.get('max_iter', 1000)
        self._knots = None
        self.spline = None
        self.basis_matrix = None
        self.coeffs = None
        self.spline = None
        self.n_features = None
        self.scaler = None
        self._intercept = None
        self._individual_feature_coeffs = None
        self.scalers_ = dict()

    def fit(self, data, targets):

        assert isinstance(data, np.ndarray), 'Data is not of type numpy.ndarray'
        if data.ndim == 2:
            self.n_features = data.shape[1]
        else:
            self.n_features = 1

        scaled_data = self._scale_transform(data)
        self._create_knots_dict(data, self.num_percentiles)

        data_basis_expansion = self._flatten_basis_for_fitting(scaled_data)

        self.coeffs = self._get_initial_coeffs(scaled_data.shape[0])
        y = targets.tolist()

        # X = np.vstack((data_basis_expansion, np.sqrt(penalty) * self._penalty_matrix()))
        # y = np.asarray(targets + np.zeros((self.num_percentiles + 2, 1)).flatten().tolist())

        norm = 0.0
        old_norm = 1.0
        idx = 0

        start = time.time()
        while (np.abs(norm - old_norm) > self.tol * norm) and (idx < self.max_iter):

            eta = np.dot(data_basis_expansion, self.coeffs)

            mu = sigmoid(eta)

            # calculate residuals
            z = _residuals(y, mu, eta)

            self.spline = sm.OLS(z, data_basis_expansion).fit()

            self.coeffs = self.spline.params

            # hat_matrix_trace = self.spline.get_influence().hat_matrix_diag[:n].sum()

            old_norm = norm
            norm = np.sum((y - sigmoid(self.spline.predict(data_basis_expansion))) ** 2)

            sys.stdout.write("\r>> Iteration: {0:04d}, elapsed time: {1:4.1f} m, norm: {2:4.1f}".format(idx + 1, (time.time() - start) / 60, norm))
            sys.stdout.flush()

            idx += 1
        sys.stdout.write('\n')
        sys.stdout.flush()

        self._intercept = self.coeffs[0]

        # Note to get the regression coeefs we need to account for the individual
        # feature of the attribute value itself in addition to the pentiles.
        self._individual_feature_coeffs = self.coeffs[1:].reshape((self.n_features, self.num_percentiles + 1))


    def _get_basis_for_array(self, array):
        return np.asarray([_get_basis_vector(array[:, idx],
                                             self._knots[idx],
                                             with_intercept=False).transpose() for idx in range(array.shape[1])]).transpose()

    def _create_knots_dict(self, data, num_percentiles):
        self._knots = {dim_idx: _get_percentiles(data[:, dim_idx], num_percentiles=num_percentiles) for dim_idx in range(data.shape[1])}

    def _flatten_basis_for_fitting(self, array):
        # since we need to fix the intercept degree of freedom we add the intercept term manually and get the individual
        # basis expansion without the intercept
        basis_expansion = self._get_basis_for_array(array)

        flattened_basis = np.ones((basis_expansion.shape[0], 1))

        for idx in range(basis_expansion.shape[2]):
            flattened_basis = np.append(flattened_basis, basis_expansion[:, :, idx], axis=1)
        return flattened_basis

    def _scale_transform(self, data):
        transformed_data = data.copy()
        for dim_idx in range(self.n_features):
            scaler = _MinMaxScaler('dim_' + str(dim_idx))
            transformed_data[:, dim_idx] = scaler.fit_transform(data[:, dim_idx])
            self.scalers_.update({dim_idx: scaler})

        return transformed_data

    def _get_basis_vector(self, vals):
        if isinstance(vals, float):
            return np.asarray([1, vals] + R(vals, self._knots).tolist())
        else:
            return np.asarray([[1, val] + R(val, self._knots).tolist() for val in vals])

    def _get_shape(self, feature_idx, vals):
        scaler = self.scalers_.get(feature_idx)
        scaled_vals = scaler.transform(vals)

        basis_expansion = np.asarray([_get_basis_vector(scaled_vals, self._knots[feature_idx], with_intercept=True)]).squeeze()
        feature_coeffs = np.asarray([self._intercept / self.n_features] + self._individual_feature_coeffs[feature_idx, :].tolist())
        return np.dot(basis_expansion, feature_coeffs)

    def _get_initial_coeffs(self, n_samples):
        coeffs = np.zeros(((self.num_percentiles + 1) * self.n_features + 1, )).flatten()
        coeffs[0] = 1 / (n_samples * ((self.num_percentiles + 1) * self.n_features + 1))
        return coeffs

    def _penalty_matrix(self):
        S = np.zeros((self.num_percentiles + 2, self.num_percentiles + 2))
        S[2:, 2:] = np.real_if_close(sp.linalg.sqrtm(R.outer(self._knots, self._knots).astype(np.float64)), tol=10 ** 8)
        return S

    def predict(self, data):
        scaled_data = self._scale_transform(data)
        return sigmoid(self.spline.predict(self._flatten_basis_for_fitting(scaled_data)))


def _calculate_residuals(y, mu, eta):
    # return (y - mu) / mu #+ eta
    if y > 0:
        ratio = np.exp(np.log(y) - np.log(mu))
    else:
        ratio = 0.0
    return ratio - 1 #+ eta

_residuals = np.frompyfunc(_calculate_residuals, 3, 1)


class _MinMaxScaler(object):

    def __init__(self, name):
        self.name = name
        self._range_min = -1.0
        self._range_max = 1.0
        self.scale = None
        self.min_val = None
        self.max_val = None

    def fit(self, values):
        data = np.asarray(values, dtype=float)

        self.min_val = data.min()
        self.max_val = data.max()

        self.scale = (self.max_val - self.min_val) / (self._range_max - self._range_min)
        return self

    def transform(self, values):
        data = np.asarray(values, dtype=float)

        return data / self.scale + (self._range_min - self.min_val / self.scale)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)
