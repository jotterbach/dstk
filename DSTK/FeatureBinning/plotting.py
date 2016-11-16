import matplotlib.pyplot as plt
import numpy as np


def plot_binner(binner, class_labels={0: '0', 1: '1'}):
    _plot_bucket_values(binner.splits, binner.values, title=binner.name, class_labels=class_labels)


def _plot_bucket_values(splits, values, title=None, class_labels={0: '0', 1: '1'}):
    class_0 = [val[0] for val in values]
    class_1 = [val[1] for val in values]
    sp = np.asarray(splits)
    non_na = sp[~np.isnan(sp)]
    non_na = np.insert(non_na, 0, np.NINF)
    label = ['({0:6.2f}, {1:6.2f}]'.format(tup[0], tup[1]) for tup in zip(non_na[:-1], non_na[1:])] + ['nan']
    ind = np.arange(len(class_0))
    w = 0.5
    plt.bar(ind, class_0, w, label=class_labels[0])
    plt.bar(ind, class_1, w, bottom=class_0, color='g', label=class_labels[1])
    plt.xticks(ind + w / 2., label, size=16, rotation=75)
    plt.yticks(size=16)
    plt.legend(fontsize=16)
    if title:
        plt.title(title, size=16)
    plt.xlabel('bucket', size=18)
    plt.ylabel('bucket value', size=18)
