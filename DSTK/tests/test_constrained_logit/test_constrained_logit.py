from __future__ import division

import sklearn.datasets as ds
from DSTK.ProjectedGradientDescentLogit import ProjectedGradientDescentLogit

import numpy as np

from sklearn.metrics import accuracy_score

cancer_ds = ds.load_breast_cancer()
data = cancer_ds['data']
target = 1-cancer_ds['target']

means = data.mean(axis=0)
stds = data.std(axis=0)
scaled_data = (data - means) / stds


def test_unconstrained_logit():
    clr = ProjectedGradientDescentLogit(data.shape[1], l1=0.0)
    clr.train(scaled_data, target, 1000, learning_rate=1e-2)

    np.testing.assert_almost_equal(accuracy_score(target, np.round(clr.predict_proba(scaled_data).squeeze())), 0.978, decimal=2)


def test_unconstrained_logit_proba_hist():
    clr = ProjectedGradientDescentLogit(data.shape[1], l1=0.0)
    clr.train(scaled_data, target, 1000, learning_rate=1e-2)

    hist, bins = np.histogram(clr.predict_proba(scaled_data).squeeze(), bins=20)
    np.testing.assert_allclose(hist, [237, 50, 20, 19, 11, 5, 4, 9, 2, 10,  3, 5, 5, 3, 2, 5, 7, 11, 18, 143], atol=15)

    np.testing.assert_almost_equal(bins, [0.00022389958030544221, 0.05021270460129017, 0.10020150962227489, 0.1501903146432596, 0.20017911966424434, 0.2501679246852291, 0.30015672970621377, 0.3501455347271985, 0.40013433974818324, 0.450123144769168, 0.5001119497901527, 0.5501007548111374, 0.6000895598321221, 0.6500783648531069, 0.7000671698740916, 0.7500559748950764, 0.800044779916061, 0.8500335849370457, 0.9000223899580305, 0.9500111949790152, 1.0], decimal=2)
