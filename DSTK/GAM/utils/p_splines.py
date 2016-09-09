from __future__ import division
import scipy as sp
import numpy as np
from statsmodels import api as sm


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


class PSpline(object):

    def __init__(self, num_percentiles=10):
        self.num_percentiles = num_percentiles
        self._knots = None
        self.spline = None
        self.basis_matrix = None

    def fit(self, values, targets, penalty=0.0):
        self._knots = _get_percentiles(values, num_percentiles=self.num_percentiles)
        self.basis_matrix = self._get_basis_vector(values)

        X = np.vstack((self.basis_matrix, np.sqrt(penalty) * self._penalty_matrix()))
        y = np.asarray(targets + np.zeros((self.num_percentiles + 2, 1)).flatten().tolist())

        self.spline = sm.OLS(y, X).fit()
        return self

    def _get_basis_vector(self, vals):
        if isinstance(vals, float):
            return np.asarray([1, vals] + R(vals, self._knots).tolist())
        else:
            return np.asarray([[1, val] + R(val, self._knots).tolist() for val in vals])

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

    def fit(self, values, targets, penalty=0.0):
        self._knots = _get_percentiles(values, num_percentiles=self.num_percentiles)
        self.basis_matrix = self._get_basis_vector(values)
        self.coeffs = self._get_initial_coeffs(len(values))

        X = np.vstack((self.basis_matrix, np.sqrt(penalty) * self._penalty_matrix()))
        y = np.asarray(targets + np.zeros((self.num_percentiles + 2, 1)).flatten().tolist())

        norm = 0.0
        old_norm = 1.0
        idx = 0
        while (np.abs(norm - old_norm) > self.tol * norm) and (idx < self.max_iter):

            eta = np.dot(X, self.coeffs)
            mu = sigmoid(eta)

            # calculate residuals
            z = (y - mu) / mu + eta

            self.spline = sm.OLS(z, X).fit()

            self.coeffs = self.spline.params
            # hat_matrix_trace = self.spline.get_influence().hat_matrix_diag[:n].sum()

            old_norm = norm
            norm = np.sum((z[:len(values)] - self.spline.predict(self.basis_matrix))**2)

            idx += 1

        print "Num iterations: ", idx

    def _get_basis_vector(self, vals):
        if isinstance(vals, float):
            return np.asarray([1, vals] + R(vals, self._knots).tolist())
        else:
            return np.asarray([[1, val] + R(val, self._knots).tolist() for val in vals])

    def _get_initial_coeffs(self, n_samples):
        coeffs = np.zeros((self.num_percentiles + 2, )).flatten()
        coeffs[0] = 1 / (n_samples * (self.num_percentiles + 2))
        return coeffs

    def _penalty_matrix(self):
        S = np.zeros((self.num_percentiles + 2, self.num_percentiles + 2))
        S[2:, 2:] = np.real_if_close(sp.linalg.sqrtm(R.outer(self._knots, self._knots).astype(np.float64)), tol=10 ** 8)
        return S

    def predict(self, values):
        return sigmoid(self.spline.predict(self._get_basis_vector(values)))
