from __future__ import division
import numpy as np
import pandas
import json as js
import tarfile as tf
import os
from DSTK.GAM.utils.shape_function import ShapeFunction
from DSTK.utils.function_helpers import sigmoid
import abc


class BaseGAM(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.shapes = dict()
        self.is_fit = False
        self.initialized = False
        self.n_features = None
        self.feature_names = None

    def _get_index_for_feature(self, feature_name):
        return self.feature_names.index(feature_name)

    def logit_score(self, vec):
        return np.sum([func.get_value(vec[self._get_index_for_feature(feat)]) for feat, func in self.shapes.iteritems()])

    def _score_single_record(self, vec):
        return [sigmoid(-2 * np.sum([func.get_value(vec[self._get_index_for_feature(feat)]) for feat, func in self.shapes.iteritems()])), \
                sigmoid( 2 * np.sum([func.get_value(vec[self._get_index_for_feature(feat)]) for feat, func in self.shapes.iteritems()]))]

    def _get_feature_value_pair_single_record(self, vec):
        return sorted([(feat, func.get_value(vec[self._get_index_for_feature(feat)])) for feat, func in self.shapes.iteritems()], key=lambda x: x[1])

    def feature_value_pairs(self, X):
        if isinstance(X, pandas.core.frame.DataFrame):
            data = X.as_matrix()
        else:
            data = X

        if data.ndim == 1:
            return self._get_feature_value_pair_single_record(data)
        if data.ndim == 2:
            return [self._get_feature_value_pair_single_record(vec) for vec in data]

    def score(self, X):
        if isinstance(X, pandas.core.frame.DataFrame):
            data = X.as_matrix()
        else:
            data = X

        if data.ndim == 1:
            return self._score_single_record(data)
        if data.ndim == 2:
            return np.asarray([self._score_single_record(vec) for vec in data])

    @staticmethod
    def _check_input(data, labels):
        if not np.isfinite(data).ravel().any():
            raise ValueError("Encountered invalid value in the training data")
        if not np.isfinite(labels).any():
            raise ValueError("Encountered invalid value in the target data")

        assert len(data) == len(labels), "Data and Targets have different lentgth."

    @abc.abstractmethod
    def train(self, data, targets, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_metadata_dict(self):
        raise NotImplementedError

    def serialize(self, model_name, file_path=None):

        if not self.is_fit:
            raise RuntimeError("Cannot serialize untrained GAM.")

        if file_path is None:
            file_path = os.getcwd()

        folder_path = '{}/{}'.format(file_path, model_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for shape in self.shapes.itervalues():
            shape.serialize('{}/{}/shapes'.format(file_path, model_name), meta_data={'model_name': model_name})

        meta_data_dct = self._get_metadata_dict()

        dct = {
            'model_name': model_name,
            'num_features': self.n_features,
            'feature_names': self.feature_names,
            'metadata': meta_data_dct,
            'shape_data': './shapes'
            }

        with open('{}/{}/{}.json'.format(file_path, model_name, model_name), 'w') as fp:
            js.dump(dct, fp, sort_keys=True, indent=2, separators=(',', ': '))

        self._make_tarfile('{}/{}.tar.gz'.format(file_path, model_name), '{}/{}'.format(file_path, model_name))

    @staticmethod
    def _make_tarfile(output_filename, source_dir):
        with tf.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))


def load_from_tar(file_name):

    model_name = file_name.split('/')[-1].replace('.tar.gz', '')

    gam = DeserializedGAM()

    with tf.open(file_name, "r:gz") as tar:
        f = tar.extractfile('{}/{}.json'.format(model_name, model_name))
        content = js.loads(f.read())
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


class DeserializedGAM(BaseGAM):
    def _get_metadata_dict(self):
        pass

    def train(self, data, targets, **kwargs):
        pass

    def __init__(self, **kwargs):
        super(DeserializedGAM, self).__init__()
