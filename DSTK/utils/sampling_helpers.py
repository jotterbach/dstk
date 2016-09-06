import numpy as np
from collections import Counter


def get_minority_majority_keys(sample_size_dict):
    return min(sample_size_dict, key=sample_size_dict.get), max(sample_size_dict, key=sample_size_dict.get)


def upsample_minority_class(data, labels, random_seed=None):

    if random_seed:
        np.random.seed(random_seed)

    cntr = Counter(labels)
    minority_key, majority_key = get_minority_majority_keys(cntr)

    minority_idx = np.where(labels == minority_key)[0]
    upsample_index = np.random.choice(minority_idx, size=cntr.get(majority_key) - cntr.get(minority_key))
    majority_idx = np.where(labels == majority_key)[0]

    upsampled_data = data[list(minority_idx) + list(upsample_index) + list(majority_idx), :]
    upsampled_labels = labels[list(minority_idx) + list(upsample_index) + list(majority_idx)]

    randomized_idx = np.random.permutation(len(upsampled_labels))

    return upsampled_data[randomized_idx, :], upsampled_labels[randomized_idx]


def downsample_majority_class(data, labels, random_seed=None):

    if random_seed:
        np.random.seed(random_seed)

    cntr = Counter(labels)
    minority_key, majority_key = get_minority_majority_keys(cntr)

    minority_idx = np.where(labels == minority_key)[0]
    majority_idx = np.where(labels == majority_key)[0]
    downsample_index = np.random.choice(majority_idx, size=cntr.get(minority_key))

    downsample_data = data[list(minority_idx) + list(downsample_index), :]
    downsample_labels = labels[list(minority_idx) + list(downsample_index)]

    return downsample_data, downsample_labels


def random_sample(data, label, sample_fraction, random_seed=None):

    if random_seed:
        np.random.seed(random_seed)

    if sample_fraction < 1.0:
        idx = int(sample_fraction * data.shape[0])
        indices = np.random.permutation(data.shape[0])
        training_idx, test_idx = indices[:idx], indices[idx:]
        x_train, x_test = data[training_idx, :], data[test_idx, :]
        y_train, y_test = label[training_idx], label[test_idx]

    else:
        x_train, x_test = data, data
        y_train, y_test = label, label

    return x_train, x_test, y_train, y_test


def create_bags(data, label, sample_fraction, num_bags, bagging_fraction, random_seed=None):

    if random_seed:
        np.random.seed(random_seed)

    x_train, x_test, y_train, y_test = random_sample(data, label, sample_fraction, random_seed=random_seed)

    bag_size = int(bagging_fraction * x_train.shape[0])
    bags = [np.random.permutation(x_train.shape[0])[:bag_size] for bag_idx in range(num_bags)]
    return x_train, x_test, y_train, y_test, bags


def permute_column_of_numpy_array(array, col_idx, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)

    perm_array = array.copy()

    perm_array[:, col_idx] = np.random.permutation(perm_array[:, col_idx])
    return perm_array
