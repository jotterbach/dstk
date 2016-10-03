from DSTK.FeatureBinning import TreeBasedFeatureBinning as tfb
import matplotlib.pyplot as plt
import numpy as np


def plot_binner(binner, class_labels={0: '0', 1: '1'}):
    assert isinstance(binner, tfb.TreeBasedFeatureBinning)

    class_0 = [tup[1][0] for tup in binner.cond_proba_buckets]
    class_1 = [tup[1][1] for tup in binner.cond_proba_buckets]
    label = [str(tup[0]).replace(')',']') for tup in binner.cond_proba_buckets]
    ind = np.arange(len(class_0))
    w = 0.5
    plt.bar(ind, class_0, w, label=class_labels[0])
    plt.bar(ind, class_1, w, bottom=class_0, color='g', label=class_labels[1])
    plt.xticks(ind + w/2., label, size=16, rotation=75)
    plt.yticks(size=16)
    plt.legend(fontsize=16)
    plt.title(binner.name, size=16)
    plt.xlabel('bucket', size=18)
    plt.ylabel('cond. proba.', size=18)
