from __future__ import division
import scipy as sp
import numpy as np
from statsmodels import api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.preprocessing import StandardScaler, MinMaxScaler


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


def _logit(x):
    if (x < 0) or (x > 1):
        raise ValueError("Value needs to be in interval [0, 1]")


    # if x >= 0.5:
    return np.log(x / (1-x))
    # elif x >
    #     return -1 * np.log((1 - x) / x)

logit = np.frompyfunc(_logit, 1, 1)


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


def _get_basis_for_array(array, num_percentiles=10, with_intercept=False):
    return np.asarray([_get_basis_vector(array[:, idx], _get_percentiles(array[:, idx], num_percentiles=num_percentiles), with_intercept=with_intercept).transpose() for idx in range(array.shape[1])]).transpose()


def _flatten_basis_for_fitting(array, num_percentiles=10):
    # since we need to fix the intercept degree of freedom we add the intercept term manually and get the individual
    # basis expansion without the intercept
    basis_expansion = _get_basis_for_array(array, num_percentiles=num_percentiles, with_intercept=False)

    flattened_basis = np.ones((basis_expansion.shape[0], 1))

    for idx in range(basis_expansion.shape[2]):
        flattened_basis = np.append(flattened_basis, basis_expansion[:, :, idx], axis=1)
    return flattened_basis


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

    def fit(self, data, targets, penalty=0.0):

        assert isinstance(data, np.ndarray), 'Data is not of type numpy.ndarray'
        if data.ndim == 2:
            self.n_features = data.shape[1]
        else:
            self.n_features = 1

        data_basis_expansion = _flatten_basis_for_fitting(data, num_percentiles=self.num_percentiles)

        # self._knots = _get_percentiles(data, num_percentiles=self.num_percentiles)
        # self.basis_matrix = self._get_basis_vector(data)
        self.coeffs = self._get_initial_coeffs(data.shape[0])

        X = data_basis_expansion
        y = targets.tolist()

        print self.n_features, data_basis_expansion.shape
        print np.min(self.coeffs), np.max(self.coeffs)
        print np.min(X), np.max(X)

        # X = np.vstack((data_basis_expansion, np.sqrt(penalty) * self._penalty_matrix()))
        # y = np.asarray(targets + np.zeros((self.num_percentiles + 2, 1)).flatten().tolist())

        norm = 0.0
        old_norm = 1.0
        idx = 0

        self.spline = Lasso(fit_intercept=True, normalize=False, alpha=penalty)
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # scaler = StandardScaler()

        scaled_basis = X #scaler.fit_transform(X)

        print np.min(scaled_basis), np.max(scaled_basis)

        # print np.std(scaled_basis, axis=0)

        while (np.abs(norm - old_norm) > self.tol * norm) and (idx < self.max_iter):

            eta = np.dot(scaled_basis, self.coeffs)

            # print 'eta: ', np.min(eta), np.max(eta)

            mu = sigmoid(eta)
            # print 'mu: ', np.min(mu), np.max(mu)

            # calculate residuals
            z = (y - mu) / mu + eta

            # print 'z: ', np.min(z), np.max(z)

            self.spline.fit(scaled_basis, z)

            # self.coeffs[0] = self.spline.intercept_
            self.coeffs = self.spline.coef_

            # print 'coeffs: ', np.min(self.coeffs), np.max(self.coeffs)
            # hat_matrix_trace = self.spline.get_influence().hat_matrix_diag[:n].sum()

            old_norm = norm
            norm = np.sum((z[:data.shape[0]] - self.spline.predict(scaled_basis)) ** 2)

            print 'norm: ', norm

            idx += 1

        print "Num iterations: ", idx

    def _get_basis_vector(self, vals):
        if isinstance(vals, float):
            return np.asarray([1, vals] + R(vals, self._knots).tolist())
        else:
            return np.asarray([[1, val] + R(val, self._knots).tolist() for val in vals])

    def _get_initial_coeffs(self, n_samples):
        coeffs = np.zeros(((self.num_percentiles + 1) * self.n_features + 1, )).flatten()
        coeffs[0] = 1 / (n_samples * ((self.num_percentiles + 1) * self.n_features + 1))
        return coeffs

    def _penalty_matrix(self):
        S = np.zeros((self.num_percentiles + 2, self.num_percentiles + 2))
        S[2:, 2:] = np.real_if_close(sp.linalg.sqrtm(R.outer(self._knots, self._knots).astype(np.float64)), tol=10 ** 8)
        return S

    def predict(self, data):
        return sigmoid(self.spline.predict(_flatten_basis_for_fitting(data, num_percentiles=self.num_percentiles)))
