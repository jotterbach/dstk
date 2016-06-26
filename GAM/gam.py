from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED
import numpy as np
import bisect


class ShapeFunction(object):

    def __init__(self, list_of_splits, list_of_values):
        assert len(list_of_splits) == len(list_of_values), 'splits and values need to be of the same length'
        assert all(list_of_splits[i] <= list_of_splits[i+1] for i in xrange(len(list_of_splits)-1)), 'range of splits has to be sorted!'

        self.splits = np.asarray(list_of_splits)
        self.values = np.asarray(list_of_values)

    def get_value(self, feature_value):
        idx = bisect.bisect(self.splits, feature_value)
        if idx == len(self.splits):
            idx = -1
        return self.values[idx]

    def multiply_by_const(self, const):
        return ShapeFunction(self.splits, const * self.values)

    def add(self, other):

        assert isinstance(other, ShapeFunction), "Can only add other shape function"

        new_splits = self.splits
        new_vals = self.values

        for split, val in zip(other.splits, other.values):
            # print 'lengths: ', len(new_splits), len(new_vals)
            idx = bisect.bisect(new_splits, split)
            new_val = val
            # print 'here'
            # print 'here {}'.format(split)
            if split in new_splits:
                # if np.isposinf(split):
                #     print split, new_val
                idx_2 = np.argwhere(new_splits == split)
                new_vals[idx_2] = new_vals[idx_2] + new_val
            elif idx == len(new_splits) and (~np.isposinf(split)):
                new_splits = np.append(new_splits, split)
                new_vals = np.append(new_vals, new_val)
            elif np.isposinf(split):
                # print 'here'
                # new_splits = np.append(new_splits, split)
                # print new_val
                new_vals[-1] = new_vals[-1] + new_val
                # new_vals = np.append(new_vals, new_val)
            else:
                new_splits = np.insert(new_splits, idx, split)
                new_vals = np.insert(new_vals, idx, new_val)

        # print new_splits
        return ShapeFunction(new_splits, new_vals)

    def __str__(self):
        return ''.join(['< {} : {}\n'.format(tup[0], tup[1]) for tup in zip(self.splits, self.values)])


class GAM(object):

    def __init__(self):
        self.shapes = dict()

    def _process(self, tree, node_id):
        print node_id, tree.feature[node_id], tree.value[node_id], tree.impurity[node_id],  tree.n_node_samples[node_id]
        return


    def recurse(self, tree, node_id):
        left_tree = tree.children_left[node_id]
        right_tree = tree.children_right[node_id]

        if left_tree == TREE_LEAF:
            return self._process(tree, node_id)
        else:
            self.recurse(tree, left_tree)

        if right_tree == TREE_LEAF:
            return self._process(tree, node_id)
        else:
            self.recurse(tree, right_tree)


    def _recurse(self, tree, feature_vec):
        break_idx = 0
        node_id = 0

        if not isinstance(feature_vec, list):
            feature_vec = list([feature_vec])

        leaf_node_id = 0
        lower = np.NINF
        upper = np.PINF

        while (node_id != TREE_LEAF) & (tree.feature[node_id] != TREE_UNDEFINED) & (break_idx <= 50):
            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]

            if feature_vec[feature_idx] <= threshold:
                upper = threshold
                if (tree.children_left[node_id] != TREE_LEAF) & (tree.children_left[node_id] != TREE_UNDEFINED):
                    node_id = tree.children_left[node_id]
                leaf_node_id = node_id
            else:
                lower = threshold
                if (tree.children_right[node_id] == TREE_LEAF) & (tree.children_right[node_id] != TREE_UNDEFINED):
                    node_id = tree.children_right[node_id]
                leaf_node_id = node_id

            break_idx += 1

        return leaf_node_id, lower, upper

    def _get_sum_of_gamma_correction(self, tree, data, labels):

        num_of_samples = {}
        sum_of_labels = {}
        weighted_sum_of_labels = {}
        set_of_boundaries = set()

        for vec, label in zip(data, labels):
            node_id, lower, upper = self._recurse(tree, vec)

            if node_id in sum_of_labels.keys():
                num_of_samples[node_id] += 1
                sum_of_labels[node_id] += label
                weighted_sum_of_labels[node_id] += np.abs(label) * (2-np.abs(label))
            else:
                num_of_samples[node_id] = 1
                sum_of_labels[node_id] = label
                weighted_sum_of_labels[node_id] = np.abs(label) * (2-np.abs(label))

            set_of_boundaries.add((node_id, lower, upper))

        lst_of_sorted_boundaries = sorted(set_of_boundaries, key=lambda x: x[1])
        split_values = [tup[2] for tup in lst_of_sorted_boundaries]
        node_keys = [tup[0] for tup in lst_of_sorted_boundaries]
        values = [(sum_of_labels[key]) / float(weighted_sum_of_labels[key]) for key in node_keys]
        # print values
        func = ShapeFunction(split_values, values)
        return func

    def _get_shape_for_attribute(self, attribute_data, labels):
        dtc = DecisionTreeRegressor(max_depth=None, max_leaf_nodes=10, min_samples_leaf=50)
        # print labels
        dtc.fit(attribute_data.reshape(-1, 1), labels)
        return self._get_sum_of_gamma_correction(dtc.tree_, attribute_data, labels)

    def _calculate_feat_shapes(self, data, labels):
        feat_shape = dict()
        for dim in range(data.shape[1]):
            func = self._get_shape_for_attribute(data[:, dim], labels)
            feat_shape[dim] = func

        return feat_shape

    def logit_score(self, vec):
        logit_val = np.sum([func.get_value(vec[key]) for key, func in self.shapes.iteritems()])
        # print logit_val
        return logit_val

    def score(self, vec):
        return 1. / (1 + np.exp( 1 * np.sum([func.get_value(vec[key]) for key, func in self.shapes.iteritems()]))),\
               1. / (1 + np.exp(-1 * np.sum([func.get_value(vec[key]) for key, func in self.shapes.iteritems()])))

    def _train_cost(self, data, labels):
        pred_labels = list()
        for vec in data:
            pred_labels.append(np.argmax(self.score(vec)))

        pred_labels = np.asarray(pred_labels)

        return np.mean(np.abs(pred_labels - (labels + 1)/2))


    def train(self, data, labels, n_iter=10, leaning_rate=0.01, display_step=25):
        val_counts = np.bincount((labels + 1) / 2)
        # print 0.5 * np.log10(val_counts[1] / float(val_counts[0]))
        self.shapes = {dim: ShapeFunction([np.PINF], [0.5 * np.log10(val_counts[1] / float(val_counts[0]))]) for dim in range(data.shape[1])}

        for epoch in range(n_iter):

            initial_labels = list()

            for vec, label in zip(data, labels):
                score = self.logit_score(vec)
                boosted_label = 2 * label / float(1 + np.exp(2 * label * score))
                # print boosted_label
                initial_labels.append(boosted_label)

            # print initial_labels

            new_shapes = self._calculate_feat_shapes(data, initial_labels)

            for dim, shape in self.shapes.iteritems():
                # if dim==0:
                #     print "dim: ", dim, "\ncurrent_shape:\n", shape.values, "new_shape:\n", new_shapes[dim].values
                self.shapes[dim] = shape.add(new_shapes[dim].multiply_by_const(leaning_rate))

            if (epoch + 1) % display_step == 0:
                print "Epoch:", '{0:04d} / {1:04d}'.format(epoch + 1, n_iter)
                print "cost: {}\n".format(self._train_cost(data, labels))

        # for dim, shape in self.shapes.iteritems():
        #     print "\ndim: {}".format(dim)
        #     print shape