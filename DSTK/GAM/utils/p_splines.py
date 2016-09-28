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
        return self.spline.predict(_get_basis_vector(values, self._knots))
