from __future__ import division
import numpy as np
from datetime import datetime
import json as js
import os
import re


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

        new_splits = self.splits.copy()
        new_vals = self.values.copy()

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