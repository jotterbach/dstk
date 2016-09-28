from __future__ import division
import scipy as sp
import numpy as np
from statsmodels import api as sm
import time
import sys
from DSTK.GAM.utils.p_splines import _get_percentiles, _get_basis_vector, R
from DSTK.utils.function_helpers import sigmoid
from DSTK.GAM.gam import ShapeFunction
from DSTK.GAM.base_gam import BaseGAM
import pandas
from datetime import datetime
from DSTK.utils import sampling_helpers as sh


def _calculate_residuals(y, mu, eta):
    # return (y - mu) / mu #+ eta
    if y > 0:
        ratio = np.exp(np.log(y) - np.log(mu))
    else:
        ratio = 0.0
    return ratio - 1 #+ eta

_residuals = np.frompyfunc(_calculate_residuals, 3, 1)


class PSplineGAM(BaseGAM):
    """
    This class implements learning a cubic spline interpolation for a binary classification problem. The implementation
    is based on the P-IRLS algorithm as described in the book:

        Simon N. Woods, Generalized Additive Models, Chapman and Hall/CRC (2006)
    """

    def _get_metadata_dict(self):
        return {
            'serialization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S%z')
        }

    def __init__(self, **kwargs):
        super(PSplineGAM, self).__init__()
        self.num_percentiles = kwargs.get('num_percentiles', 10)
        self.tol = kwargs.get('tol', 5e-4)
        self.max_iter = kwargs.get('max_iter', 1000)
        self.balancer = kwargs.get('balancer', None)
        self.balancer_seed = kwargs.get('balancer_seed', None)
        self._knots = None
        self.spline = None
        self.basis_matrix = None
        self.coeffs = None
        self.spline = None
        self.scaler = None
        self._intercept = None
        self._individual_feature_coeffs = None
        self.scalers_ = dict()

        _allowed_balancers = ['global_upsample', 'global_downsample']

        if not self.balancer is None:
            if not self.balancer in _allowed_balancers:
                raise NotImplementedError(
                    "Balancing method '{}' not implemented. Choose one of {}.".format(self.balancer,
                                                                                      _allowed_balancers))

    def _init_properties(self, data, labels):
        self.n_features = data.shape[1]

        if isinstance(data, pandas.core.frame.DataFrame):
            self.feature_names = data.columns.tolist()
            data = data.as_matrix()

        if self.feature_names is None:
            self.feature_names = ['feature_{}'.format(dim) for dim in range(self.n_features)]

        if isinstance(labels, pandas.core.series.Series):
            labels = labels.values

        return data, labels

    def train(self, data, targets, **kwargs):

        penalty = kwargs.get('penalty', None)

        data, labels = self._init_properties(data, targets)

        assert isinstance(data, np.ndarray), 'Data is not of type numpy.ndarray'
        if data.ndim == 2:
            self.n_features = data.shape[1]
        else:
            self.n_features = 1

        if self.balancer == 'global_upsample':
            data, targets = sh.upsample_minority_class(data, targets, random_seed=self.balancer_seed)
        elif self.balancer == 'global_downsample':
            data, targets = sh.downsample_majority_class(data, targets, random_seed=self.balancer_seed)

        scaled_data = self._scale_transform(data)
        self._create_knots_dict(data, self.num_percentiles)

        data_basis_expansion = self._flatten_basis_for_fitting(scaled_data)

        self.coeffs = self._get_initial_coeffs(scaled_data.shape[0])

        if penalty:
            data_basis_expansion = np.vstack((data_basis_expansion, np.sqrt(penalty) * self._penalty_matrix()))
            y = np.asarray(targets.tolist() + np.zeros(((self.num_percentiles + 1) * self.n_features + 1, 1)).squeeze().tolist()).flatten().squeeze()
        else:
            y = np.asarray(targets.tolist())

        norm = 0.0
        old_norm = 1.0
        idx = 0

        start = time.time()
        while (np.abs(norm - old_norm) > self.tol * norm) and (idx < self.max_iter):

            eta = np.dot(data_basis_expansion, self.coeffs)

            mu = sigmoid(eta)

            # calculate residuals
            z = _residuals(y, mu, eta)
            if penalty:
                z[len(targets) + 1:] = np.zeros(((self.num_percentiles + 1) * self.n_features, 1)).squeeze()

            self.spline = sm.OLS(z, data_basis_expansion).fit()

            self.coeffs = self.spline.params

            old_norm = norm
            norm = np.sum((z - self.spline.predict(data_basis_expansion)) ** 2)

            sys.stdout.write("\r>> Iteration: {0:04d}, elapsed time: {1:4.1f} m, norm: {2:4.1f}".format(idx + 1, (time.time() - start) / 60, norm))
            sys.stdout.flush()

            idx += 1
        sys.stdout.write('\n')
        sys.stdout.flush()

        self._intercept = self.coeffs[0]

        # Note to get the regression coeefs we need to account for the individual
        # feature of the attribute value itself in addition to the pentiles.
        self._individual_feature_coeffs = self.coeffs[1:].reshape((self.n_features, self.num_percentiles + 1))
        self._create_shape_functions(data)
        self.is_fit = True
        return self

    def gcv_score(self):
        X = self.spline.model.exog[:-(self.num_percentiles + 2), :]
        n = X.shape[0]
        y = self.spline.model.endog[:n]
        y_hat = self.spline.predict(X)

        hat_matrix_trace = self.spline.get_influence().hat_matrix_diag[:n].sum()

        return n * np.power(y - y_hat, 2).sum() / np.power(n - hat_matrix_trace, 2)

    def _get_basis_for_array(self, array):
        return np.asarray([_get_basis_vector(array[:, self._get_index_for_feature(feat)],
                                             self._knots[feat],
                                             with_intercept=False).transpose() for feat in self.feature_names]).transpose()

    def _create_knots_dict(self, data, num_percentiles):
        self._knots = {feat: _get_percentiles(data[:, self._get_index_for_feature(feat)], num_percentiles=num_percentiles) for feat in self.feature_names}

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

        basis_expansion = np.asarray([_get_basis_vector(scaled_vals, self._knots[self.feature_names[feature_idx]], with_intercept=True)]).squeeze()
        feature_coeffs = np.asarray([self._intercept / self.n_features] + self._individual_feature_coeffs[feature_idx, :].tolist())
        return np.dot(basis_expansion, feature_coeffs)

    def _create_shape_functions(self, data):
        shapes = dict()
        for feat in self.feature_names:
            dim_idx = self._get_index_for_feature(feat)
            splits = np.unique(data[:, dim_idx].flatten()).tolist()
            vals = self._get_shape(dim_idx, splits)

            shapes[feat] = ShapeFunction(splits, vals, feat)

        self.shapes = shapes

    def _get_initial_coeffs(self, n_samples):
        coeffs = np.zeros(((self.num_percentiles + 1) * self.n_features + 1, )).flatten()
        coeffs[0] = 1 / (n_samples * ((self.num_percentiles + 1) * self.n_features + 1))
        return coeffs

    def _penalty_matrix(self):
        S = np.zeros(((self.num_percentiles + 1) * self.n_features + 1, (self.num_percentiles + 1) * self.n_features + 1))

        flat_knots = np.asarray([self._knots.get(feat) for feat in self.feature_names]).flatten()

        penalty_matrix = np.real_if_close(sp.linalg.sqrtm(R.outer(flat_knots, flat_knots).astype(np.float64)), tol=10 ** 8)

        penalty_matrix = np.insert(penalty_matrix,
                                   np.arange(0, self.num_percentiles * self.n_features, self.num_percentiles),
                                   0,
                                   axis=1)
        penalty_matrix = np.insert(penalty_matrix,
                                   np.arange(0, self.num_percentiles * self.n_features, self.num_percentiles),
                                   0,
                                   axis=0)

        S[1:, 1:] = penalty_matrix
        return S

    def predict(self, data):
        scaled_data = self._scale_transform(data)
        return sigmoid(self.spline.predict(self._flatten_basis_for_fitting(scaled_data)))


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
