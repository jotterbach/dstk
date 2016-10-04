from __future__ import division
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED, Tree
from sklearn.metrics import roc_auc_score
import numpy as np
from collections import Counter
from operator import itemgetter
import pandas
from datetime import datetime
import sys
import time
import DSTK.utils.sampling_helpers as sh
import DSTK.utils.metrics as metrics
from concurrent import futures
from DSTK.GAM.utils.p_splines import PSpline
from DSTK.GAM.utils.shape_function import ShapeFunction
from DSTK.utils.function_helpers import sigmoid
from DSTK.GAM.base_gam import BaseGAM
from DSTK.FeatureBinning.TreeBasedFeatureBinning import _recurse_tree


def _flatten_tree(tree):
    list_of_buckets = list()
    _recurse_tree(tree, list_of_buckets, mdlp=True)
    lower_bounds = np.asarray([tup[0][0] for tup in list_of_buckets], dtype=np.float64, order='C')
    upper_bounds = np.asarray([tup[0][1] for tup in list_of_buckets], dtype=np.float64, order='C')
    bucket_values = np.asarray([tup[1][0] for tup in list_of_buckets], dtype=np.float64, order='C')
    return lower_bounds, upper_bounds, bucket_values


def _recurse(tree, feature_vec):

    assert isinstance(tree, Tree), "Tree is not a sklearn Tree"

    break_idx = 0
    node_id = 0

    if not isinstance(feature_vec, list):
        feature_vec = list([feature_vec])

    leaf_node_id = 0
    lower = np.NINF
    upper = np.PINF

    while (node_id != TREE_LEAF) & (tree.feature[node_id] != TREE_UNDEFINED):
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]

        if np.float32(feature_vec[feature_idx]) <= threshold:
            upper = threshold
            if (tree.children_left[node_id] != TREE_LEAF) and (tree.children_left[node_id] != TREE_UNDEFINED):
                leaf_node_id = tree.children_left[node_id]
            node_id = tree.children_left[node_id]
        else:
            lower = threshold
            if (tree.children_right[node_id] == TREE_LEAF) and (tree.children_right[node_id] != TREE_UNDEFINED):
                leaf_node_id = tree.children_right[node_id]
            node_id = tree.children_right[node_id]

        break_idx += 1
        if break_idx > 2 * tree.node_count:
            raise RuntimeError("infinite recursion!")

    return leaf_node_id, lower, upper


def _get_grouped_pseudo_responses(data, labels, upper_bounds):
    node_ids = np.digitize(data, upper_bounds, right=True)
    n_groups = np.unique(node_ids)

    return n_groups, np.asarray([labels[np.where(node_ids == idx)[0]] for idx in n_groups])


def _get_sum_of_gamma_correction(tree, data, labels, feature_name):

    lower_bounds, upper_bounds, bucket_values = _flatten_tree(tree)
    nodes, grouped_labels = _get_grouped_pseudo_responses(data, labels, upper_bounds)

    sum_of_labels = np.asarray([grouped_labels[idx].sum() for idx in nodes])
    weighted_sum_of_labels = np.asarray([np.sum(np.abs(grouped_labels[idx]) * (2 - np.abs(grouped_labels[idx]))) for idx in nodes])
    values = sum_of_labels / weighted_sum_of_labels

    if not np.isfinite(values).any():
        raise ArithmeticError("Encountered NaN or Infinity. Aborting training")

    return ShapeFunction(upper_bounds, values, feature_name)


def _get_shape_for_attribute(attribute_data, labels, feature_name, criterion, splitter,
                             max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                             max_features, random_state, max_leaf_nodes, presort):

    dtr = DecisionTreeRegressor(criterion=criterion,
                                splitter=splitter,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                max_features=max_features,
                                random_state=random_state,
                                max_leaf_nodes=max_leaf_nodes,
                                presort=presort)

    dtr.fit(attribute_data.reshape(-1, 1), labels)
    return feature_name, _get_sum_of_gamma_correction(dtr.tree_, attribute_data, labels, feature_name)


class GAM(BaseGAM):

    def __init__(self, **kwargs):
        super(GAM, self).__init__()
        self.class_weights = np.ones(2)
        self._recording = {
            'epoch': 0,
            'costs': {
                'accuracy': [],
                'precision': [],
                'prevalence': [],
                'recall': [],
                'roc_auc': []
            },
            'learning_rate_schedule': dict()
        }

        self.criterion = kwargs.get('criterion', 'mse')
        self.splitter = kwargs.get('splitter', 'best')
        self.max_depth = kwargs.get('max_depth', None)
        self.min_samples_split = kwargs.get('min_samples_split', 2)
        self.min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        self.min_weight_fraction_leaf = kwargs.get('min_weight_fraction_leaf', 0.0)
        self.max_features = kwargs.get('max_features', None)
        self.random_state = kwargs.get('random_state', None)
        self.max_leaf_nodes = kwargs.get('max_leaf_nodes', None)
        self.presort = kwargs.get('presort', False)

        self.balancer = kwargs.get('balanced', None)
        self.balancer_seed = kwargs.get('balancer_seed', None)
        self.influence_trimming_threshold = kwargs.get('influence_trimming_threshold', None)

        _allowed_balancers = ['global_upsample', 'global_downsample', 'boosted_upsample', 'boosted_downsample']

        if not self.balancer is None:
            if not self.balancer in _allowed_balancers:
                raise NotImplementedError("Balancing method '{}' not implemented. Choose one of {}.".format(self.balancer, _allowed_balancers))

    def _get_metadata_dict(self):
        return {
            'training_metadata': self._recording,
            'balancer': self.balancer,
            'DecisionTreeRegressor_meta_data': {
                'criterion': self.criterion,
                'splitter': self.splitter,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'min_weight_fraction_leaf': self.min_weight_fraction_leaf,
                'max_features': self.max_features,
                'random_state': self.random_state,
                'max_leaf_nodes': self.max_leaf_nodes,
                'presort': self.presort,
            },
            'serialization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S%z')
        }

    def _train_cost(self, data, labels):
        pred_scores = np.asarray([self.score(vec) for vec in data], dtype='float')
        pred_labels = [2 * np.argmax(score) - 1 for score in pred_scores]

        prevalence, precision, recall, accuracy, f1, prior, support = metrics.get_all_metrics(labels,
                                                                                              pred_labels,
                                                                                              neg_class_label=-1)
        self._recording['costs']['accuracy'].append(accuracy)
        self._recording['costs']['precision'].append(precision)
        self._recording['costs']['prevalence'].append(prevalence)
        self._recording['costs']['recall'].append(recall)
        self._recording['costs']['roc_auc'].append(roc_auc_score(labels, pred_scores[:, 1]))

        return accuracy,\
               precision,\
               prevalence,\
               recall,\
               roc_auc_score(labels, pred_scores[:, 1])

    def _get_pseudo_responses(self, data, labels):
        return np.asarray([2 * label * sigmoid(-2 * label * self.logit_score(vec)) for vec, label in zip(data, labels)])

    def _init_shapes_and_data(self, data, labels):

        self.n_features = data.shape[1]

        if isinstance(data, pandas.core.frame.DataFrame):
            self.feature_names = data.columns.tolist()
            data = data.as_matrix()

        if self.feature_names is None:
            self.feature_names = ['feature_{}'.format(dim) for dim in range(self.n_features)]

        if isinstance(labels, pandas.core.series.Series):
            labels = labels.values

        cntr = Counter(labels)
        assert set(cntr.keys()) == {-1, 1}, "Labels must be encoded with -1, 1. Cannot contain more classes."
        assert self.n_features is not None, "Number of attributes is None"

        self.shapes = {name: ShapeFunction([np.PINF],
                                           [0.0],
                                           name)
                       for name in self.feature_names}
        self.initialized = True

        return data, labels

    def _update_learning_rate(self, dct, epoch):

        epoch_key = max(k for k in dct if k <= epoch)
        if self._recording['epoch'] <= 1:
            self._current_lr = dct[epoch_key]
            self._recording['learning_rate_schedule'].update({self._recording['epoch'] - 1: self._current_lr})

        if dct[epoch_key] != self._current_lr:
            self._current_lr = dct[epoch_key]
            self._recording['learning_rate_schedule'].update({self._recording['epoch'] - 1: self._current_lr})

        return self._current_lr

    def _initialize_class_weights(self, labels):
        cntr = Counter(labels)
        bin_count = np.asarray([x[1] for x in sorted(cntr.items(), key=itemgetter(0))])
        self.class_weights = bin_count.sum() / (2.0 * bin_count)

    def _get_class_weights(self, labels):
        return self.class_weights[np.asarray((labels + 1)/2, dtype=int)]

    def train(self, data, targets, **kwargs):

        n_iter = kwargs.get('n_iter', 10)
        learning_rate = kwargs.get('learning_rate', 0.01)
        sample_fraction = kwargs.get('sample_fraction', 1.0)
        num_bags = kwargs.get('num_bags', 1)
        num_workers = kwargs.get('num_workers', 1)

        if not self.initialized:
            data, targets = self._init_shapes_and_data(data, targets)
        else:
            if isinstance(data, pandas.core.frame.DataFrame):
                data = data.as_matrix()
            if isinstance(targets, pandas.core.series.Series):
                targets = targets.values

        self._check_input(data, targets)

        if self.balancer == 'global_upsample':
            data, targets = sh.upsample_minority_class(data, targets, random_seed=self.balancer_seed)
        elif self.balancer == 'global_downsample':
            data, targets = sh.downsample_majority_class(data, targets, random_seed=self.balancer_seed)
        elif self.balancer == 'class_weights':
            self._initialize_class_weights(targets)

        start = time.time()
        for epoch in range(n_iter):
            self._recording['epoch'] += 1

            if isinstance(learning_rate, dict):
                lr = self._update_learning_rate(learning_rate, epoch)
            else:
                lr = learning_rate

            x_train, x_test, y_train, y_test, bags = sh.create_bags(data, targets, sample_fraction, num_bags, bagging_fraction=0.5, random_seed=self.balancer_seed)

            new_shapes = self._calculate_gradient_shape(x_train, y_train, bag_indices=bags, max_workers=num_workers)

            self.shapes = {dim: self.shapes[dim].add(shape.multiply(lr)) for dim, shape in new_shapes.iteritems()}

            acc, prec, prev, rec, auc = self._train_cost(x_test, y_test)

            sys.stdout.write("\r>> Epoch: {0:04d} / {1:04d}, elapsed time: {2:4.1f} m -- accuracy: {3:1.3f}, precision: {4:1.3f}, prevalence: {5:1.3f}, recall: {6:1.3f}, roc_auc: {7:1.3f}".format(epoch + 1, n_iter, (time.time()-start)/60, acc, prec, prev, rec, auc))
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()

        self.is_fit = True

    def _get_trimmed_record_indices(self, responses):
        weights = np.abs(responses) * (2 - np.abs(responses))
        sum_weights = weights.sum()
        sorted_idx = np.argsort(weights).flatten()
        truncate_idx = (np.cumsum(weights[sorted_idx]) <= self.influence_trimming_threshold * sum_weights).sum()
        return np.where(weights >= weights[sorted_idx[truncate_idx]])[0]

    def _calculate_gradient_shape(self, data, labels, bag_indices=None, max_workers=1):

        if bag_indices is None:
            bag_indices = range(len(labels))

        for bag_idx, bag in enumerate(bag_indices):
            x_train = data[bag, :]
            y_train = labels[bag]

            if self.balancer == 'boosted_upsample':
                x_train, y_train = sh.upsample_minority_class(x_train, y_train, random_seed=self.balancer_seed)
            elif self.balancer == 'boosted_downsample':
                x_train, y_train = sh.downsample_majority_class(x_train, y_train, random_seed=self.balancer_seed)
            responses = self._get_pseudo_responses(x_train, y_train)

            if self.influence_trimming_threshold:
                train_records_idx = self._get_trimmed_record_indices(responses)
                x_train = x_train[train_records_idx, :]
                responses = responses[train_records_idx]

            with futures.ProcessPoolExecutor(max_workers=max_workers) as executors:
                lst_of_futures = [executors.submit(_get_shape_for_attribute,
                                                   x_train[:, self._get_index_for_feature(name)],
                                                   responses,
                                                   name,
                                                   self.criterion,
                                                   self.splitter,
                                                   self.max_depth,
                                                   self.min_samples_split,
                                                   self.min_samples_leaf,
                                                   self.min_weight_fraction_leaf,
                                                   self.max_features,
                                                   self.random_state,
                                                   self.max_leaf_nodes,
                                                   self.presort) for name in self.feature_names]

                results = [f.result() for f in futures.as_completed(lst_of_futures)]

            if bag_idx == 0:
                new_shapes = {res[0]: res[1] for res in results}

            else:
                old_shapes = new_shapes.copy()
                new_shapes = {res[0]: old_shapes[res[0]].add(res[1]) for res in results}

        return {name: shape.multiply(1 / len(bag_indices)) for name, shape in new_shapes.iteritems()}


class SmoothGAM(BaseGAM):

    def __init__(self, gam):

        assert isinstance(gam, BaseGAM)

        super(SmoothGAM, self).__init__()
        self.gam = gam
        self.feature_names = gam.feature_names
        self.n_features = gam.n_features
        self.is_fit = gam.is_fit
        self.initialized = gam.initialized
        self.shapes = dict()

    def __getattr__(self, item):
        try:
            return self.gam.__getattribute__(item)
        except AttributeError:
            return self.__getattr__(item)

    def smoothen(self, data, penalty=None):
        if isinstance(penalty, np.ndarray):
            penalties = penalty.tolist()
        if penalty is None:
            penalties = [0.0]
        if isinstance(penalty, float):
            penalties = [penalty]

        for key, shape in self.gam.shapes.iteritems():
            print 'processing shape `{}`'.format(key)
            self.shapes.update({key: SmoothGAM._create_smooth_shape(shape, data[key], key, penalties)})

    @staticmethod
    def _create_smooth_shape(shape, values, name, penalties):
        if shape.splits.shape == (2, ):
            return shape
        else:
            target_vals = [shape.get_value(val) for val in values]
            result = SmoothGAM._fit_spline(values, target_vals, penalties)
            splits = np.unique(values)
            smooth_values = result.predict(splits)
            return ShapeFunction(splits, smooth_values, name)

    @staticmethod
    def _fit_spline(values, target_vals, penalties):

        spl = PSpline()
        gcv_error = [spl.fit(values, target_vals, penalty=penalty).gcv_score() for penalty in penalties]
        opt_lambda = penalties[np.argmin(gcv_error)]
        return spl.fit(values, target_vals, penalty=opt_lambda)
