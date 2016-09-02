from __future__ import division
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED, Tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from collections import Counter
from operator import itemgetter
import pandas
from datetime import datetime
import json as js
import tarfile as tf
import os
import re
import math
import sys
import time
import DSTK.utils.sampling_helpers as sh
import DSTK.utils.metrics as metrics
from concurrent import futures


def sigmoid(x):
    """
    Numerically-stable sigmoid function.
    Taken from:
      - http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
      - http://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
    """
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = math.exp(x)
        return z / (1 + z)


class ShapeFunction(object):

    def __init__(self, list_of_splits, list_of_values, name):
        assert len(list_of_splits) == len(list_of_values), 'splits and values need to be of the same length'
        assert all(list_of_splits[i] <= list_of_splits[i+1] for i in xrange(len(list_of_splits)-1)), 'range of splits has to be sorted!'

        self.splits = np.asarray(list_of_splits, dtype=np.float64)
        self.values = np.asarray(list_of_values, dtype=np.float64)
        self.name = name

    def get_value(self, feature_value):
        idx = np.searchsorted(self.splits, feature_value, side='right')
        if idx == len(self.splits):
            idx = -1
        return self.values[idx]

    def multiply(self, const):
        return ShapeFunction(self.splits, const * self.values, self.name)

    def add(self, other):
        return self.__add__(other)

    def __add__(self, other):

        assert isinstance(other, ShapeFunction), "Can only add other shape function"

        assert self.name == other.name, "Cannot add shapes of different features"

        new_splits = self.splits
        new_vals = self.values

        for split, val in zip(other.splits, other.values):
            idx = np.searchsorted(new_splits, split, side='right')
            new_val = val
            if split in new_splits:
                idx_2 = np.argwhere(new_splits == split)
                new_vals[idx_2] = new_vals[idx_2] + new_val
            elif idx == len(new_splits) and (~np.isposinf(split)):
                new_splits = np.append(new_splits, split)
                new_vals = np.append(new_vals, new_val)
            elif np.isposinf(split):
                new_vals[-1] = new_vals[-1] + new_val
            else:
                new_splits = np.insert(new_splits, idx, split)
                new_vals = np.insert(new_vals, idx, new_val)

        return ShapeFunction(new_splits, new_vals, self.name)

    def __str__(self):
        return ''.join(['< {} : {}\n'.format(tup[0], tup[1]) for tup in zip(self.splits, self.values)])

    def equals(self, other):
        return (self.splits == other.splits).all() and (self.values == other.values).all()

    def serialize(self, file_path, meta_data=dict()):
        meta_data_dct = {
            'serialization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S%z')
        }
        meta_data_dct.update(meta_data)

        dct = {
            'feature_name': self.name,
            'splits': self.splits.tolist(),
            'values': self.values.tolist(),
            'split_rule': 'LT',
            'meta_data': meta_data_dct
        }

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        with open('{}/{}.json'.format(file_path, ShapeFunction.create_clean_file_name(self.name)), 'w') as fp:
            js.dump(dct, fp, sort_keys=True, indent=2, separators=(',', ': '))


    @staticmethod
    def create_clean_file_name(file_name):
        cleaning_pattern = re.compile(r'[\\#\.\$@!><\|/]+')
        return re.sub(cleaning_pattern, "__", file_name)

    @staticmethod
    def load_from_json(file_path):
        with open(file_path, 'r') as fp:
            dct = js.load(fp=fp)

        return ShapeFunction(np.asarray(dct['splits']),
                             np.asarray(dct['values']),
                             dct['feature_name'])


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


def _get_sum_of_gamma_correction(tree, data, labels, class_weights, feature_name):

    num_of_samples = {}
    sum_of_labels = {}
    weighted_sum_of_labels = {}
    set_of_boundaries = set()

    for vec, label, weight in zip(data, labels, class_weights):
        node_id, lower, upper = _recurse(tree, vec)

        if node_id in sum_of_labels.keys():
            num_of_samples[node_id] += 1
            sum_of_labels[node_id] += weight * label
            weighted_sum_of_labels[node_id] += weight * np.abs(label) * (2 - np.abs(label))
        else:
            num_of_samples[node_id] = 1
            sum_of_labels[node_id] = weight * label
            weighted_sum_of_labels[node_id] = weight * np.abs(label) * (2 - np.abs(label))

        set_of_boundaries.add((node_id, lower, upper))

    lst_of_sorted_boundaries = sorted(set_of_boundaries, key=lambda x: x[1])
    split_values = [tup[2] for tup in lst_of_sorted_boundaries]
    node_keys = [tup[0] for tup in lst_of_sorted_boundaries]
    values = [(sum_of_labels[key]) / float(weighted_sum_of_labels[key]) for key in node_keys]

    if not np.isfinite(values).any():
        raise ArithmeticError("Encountered NaN or Infinity. Aborting training")

    return ShapeFunction(split_values, values, feature_name)


def _get_shape_for_attribute(attribute_data, labels, class_weights, feature_name, criterion, splitter,
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
    return feature_name, _get_sum_of_gamma_correction(dtr.tree_, attribute_data, labels, class_weights, feature_name)


class GAM(object):

    def __init__(self, **kwargs):
        self.shapes = dict()
        self.is_fit = False
        self._n_features = None
        self.initialized = False
        self.feature_names = None
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

        _allowed_balancers = ['global_upsample', 'global_downsample', 'boosted_upsample', 'boosted_downsample', 'class_weights']

        if not self.balancer is None:
            if not self.balancer in _allowed_balancers:
                raise NotImplementedError("Balancing method '{}' not implemented. Choose one of {}.".format(self.balancer, _allowed_balancers))

    def _get_index_for_feature(self, feature_name):
        return self.feature_names.index(feature_name)

    def logit_score(self, vec):
        return np.sum([func.get_value(vec[self._get_index_for_feature(feat)]) for feat, func in self.shapes.iteritems()])

    def score(self, vec):
        return sigmoid(-2 * np.sum([func.get_value(vec[self._get_index_for_feature(feat)]) for feat, func in self.shapes.iteritems()])),\
               sigmoid( 2 * np.sum([func.get_value(vec[self._get_index_for_feature(feat)]) for feat, func in self.shapes.iteritems()]))

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
        return [2 * label * sigmoid(-2 * label * self.logit_score(vec)) for vec, label in zip(data, labels)]

    def _init_shapes_and_data(self, data, labels):

        self._n_features = data.shape[1]

        if isinstance(data, pandas.core.frame.DataFrame):
            self.feature_names = data.columns.tolist()
            data = data.as_matrix()

        if self.feature_names is None:
            self.feature_names = ['feature_{}'.format(dim) for dim in range(self._n_features)]

        if isinstance(labels, pandas.core.series.Series):
            labels = labels.values

        cntr = Counter(labels)
        assert set(cntr.keys()) == {-1, 1}, "Labels must be encoded with -1, 1. Cannot contain more classes."
        assert self._n_features is not None, "Number of attributes is None"

        self.shapes = {name: ShapeFunction([np.PINF],
                                           [np.log(cntr.get(1, 0) / cntr.get(-1, 1)) / (2 * self._n_features)],
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

    @staticmethod
    def _check_input(data, labels):
        if not np.isfinite(data).ravel().any():
            raise ValueError("Encountered invalid value in the training data")
        if not np.isfinite(labels).any():
            raise ValueError("Encountered invalid value in the target data")

        assert len(data) == len(labels), "Data and Targets have different lentgth."

    def _initialize_class_weights(self, labels):
        cntr = Counter(labels)
        bin_count = np.asarray([x[1] for x in sorted(cntr.items(), key=itemgetter(0))])
        self.class_weights = bin_count.sum() / (2.0 * bin_count)

    def _get_class_weights(self,labels):
        return self.class_weights[np.asarray((labels + 1)/2, dtype=int)]

    def train(self, data, labels, n_iter=10, learning_rate=0.01, sample_fraction=1.0, num_bags=1, num_workers=1):
        if not self.initialized:
            data, labels = self._init_shapes_and_data(data, labels)
        else:
            if isinstance(data, pandas.core.frame.DataFrame):
                data = data.as_matrix()
            if isinstance(labels, pandas.core.series.Series):
                labels = labels.values

        self._check_input(data, labels)

        if self.balancer == 'global_upsample':
            data, labels = sh.upsample_minority_class(data, labels, random_seed=self.balancer_seed)
        elif self.balancer == 'global_downsample':
            data, labels = sh.downsample_majority_class(data, labels, random_seed=self.balancer_seed)
        elif self.balancer == 'class_weights':
            self._initialize_class_weights(labels)

        start = time.time()
        for epoch in range(n_iter):
            self._recording['epoch'] += 1

            if isinstance(learning_rate, dict):
                lr = self._update_learning_rate(learning_rate, epoch)
            else:
                lr = learning_rate

            x_train, x_test, y_train, y_test, bags = sh.create_bags(data, labels, sample_fraction, num_bags, bagging_fraction=0.5, random_seed=self.balancer_seed)

            new_shapes = self._calculate_gradient_shape(x_train, y_train, bag_indices=bags, max_workers=num_workers)

            self.shapes = {dim: self.shapes[dim].add(shape.multiply(lr)) for dim, shape in new_shapes.iteritems()}

            acc, prec, prev, rec, auc = self._train_cost(x_test, y_test)

            sys.stdout.write("\r>> Epoch: {0:04d} / {1:04d}, elapsed time: {2:4.1f} m -- accuracy: {3:1.3f}, precision: {4:1.3f}, prevalence: {5:1.3f}, recall: {6:1.3f}, roc_auc: {7:1.3f}".format(epoch + 1, n_iter, (time.time()-start)/60, acc, prec, prev, rec, auc))
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()

        self.is_fit = True

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
            class_weights = self._get_class_weights(y_train)
            responses = self._get_pseudo_responses(x_train, y_train)
            with futures.ProcessPoolExecutor(max_workers=max_workers) as executors:
                lst_of_futures = [executors.submit(_get_shape_for_attribute,
                                                   x_train[:, self._get_index_for_feature(name)],
                                                   responses,
                                                   class_weights,
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

    def serialize(self, model_name, file_path=None):

        if file_path is None:
            file_path = os.getcwd()

        folder_path = '{}/{}'.format(file_path, model_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for shape in self.shapes.itervalues():
            shape.serialize('{}/{}/shapes'.format(file_path, model_name), meta_data={'model_name': model_name})

        meta_data_dct = {
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

        dct = {
            'model_name': model_name,
            'num_features': self._n_features,
            'feature_names': self.feature_names,
            'metadata': meta_data_dct,
            'shape_data': './shapes'
            }

        with open('{}/{}/{}.json'.format(file_path, model_name, model_name), 'w') as fp:
            js.dump(dct, fp, sort_keys=True, indent=2, separators=(',', ': '))

        GAM._make_tarfile('{}/{}.tar.gz'.format(file_path, model_name), '{}/{}'.format(file_path, model_name))

    @staticmethod
    def _make_tarfile(output_filename, source_dir):
        with tf.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))

    @staticmethod
    def load_from_tar(file_name):

        model_name = file_name.split('/')[-1].replace('.tar.gz', '')

        gam = GAM()

        with tf.open(file_name, "r:gz") as tar:
            f = tar.extractfile('{}/{}.json'.format(model_name, model_name))
            content = js.loads(f.read())
            gam._recording = content['metadata']['training_metadata']
            gam._n_features = content['num_features']
            gam.feature_names = content['feature_names']

            for member in tar.getmembers():
                if member.isfile() and (member.name != '{}/{}.json'.format(model_name, model_name)):
                    f = tar.extractfile(member.path)
                    content = js.loads(f.read())
                    gam.shapes.update({content['feature_name']: ShapeFunction(content['splits'],
                                                                              content['values'],
                                                                              content['feature_name'])})

        assert set(gam.shapes.keys()) == set(gam.feature_names), 'feature names and shape names do not match up'
        return gam
