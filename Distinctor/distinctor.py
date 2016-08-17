from __future__ import division
import numpy as np
from fuzzywuzzy import fuzz
from functools import partial


def inv_luminance_ratio(val_0, val_1):
    """
    See https://en.wikipedia.org/wiki/Display_contrast
    :param val_0:
    :param val_1:
    :return:
    """
    try:
        val_0 = float(val_0)
        val_1 = float(val_1)

        min_val = np.min([np.abs(val_0), np.abs(val_1)])
        max_val = np.max([np.abs(val_0), np.abs(val_1)])
        if min_val == max_val:
            return 1.0
        if max_val > 0:
            return min_val / max_val
        else:
            return 1.0

    except TypeError:
        return 1.0


def contrast_modulation(val_0, val_1):
    """
    See https://en.wikipedia.org/wiki/Display_contrast
    :param val_0:
    :param val_1:
    :return:
    """
    inv_lum = inv_luminance_ratio(val_0, val_1)
    return (1 - inv_lum) / (1 + inv_lum)


def inv_contrast_modulation(val_0, val_1):
    """
    See https://en.wikipedia.org/wiki/Display_contrast
    :param val_0:
    :param val_1:
    :return:
    """
    inv_lum = inv_luminance_ratio(val_0, val_1)
    return (2 * inv_lum) / (1 + inv_lum)


def fuzzy_ratio(str_0, str_1, override_list=None, override_return=np.NaN):
    """
    Returns the fuzzy matching ratio between the provided strings. If one of the two string is in the `override_list` then it returns
    `override_return` value.
    :param str_0:
    :param str_1:
    :param override_list:
    :param override_return:
    :return:
    """
    if override_list and ((str(str_0).lower() in override_list) or (str(str_1).lower() in override_list)):
        return override_return
    return fuzz.ratio(str(str_0), str(str_1)) / 100


def get_fuzzy_ratio_matcher_with_overrides(override_list, override_return):
    """
    Returns a `fuzzy_ratio` matcher function with a given `override_list` and `override_return`.Useful for functional programming where we want to pre-specify
    those values.
    :param override_list:
    :param override_return:
    :return:
    """
    return partial(fuzzy_ratio, override_list=override_list, override_return=override_return)


def exponential_time_match(date_0, date_1, scale):
    """
    Returns the negative exponential of the difference between the two provided dates normalized by the scale parameter given in days.
    E.g. if the date difference is 1 day and the scale is one day, this method return e^{-1}.
    :param date_0:
    :param date_1:
    :param scale:
    :return:
    """
    return np.exp(-1.0 * np.abs((date_1 - date_0).days) / scale)


def get_exponential_time_matcher_with_scale(scale):
    """
    returns an `exponential_time_match` function with a given scale. Useful for functional programming where we want to pre-specify the scale
    :param scale:
    :return:
    """
    return partial(exponential_time_match, scale=scale)


def symmetric_max_exponential_time_match(date_0, data_1, date_2, date_3, scale):
    """
    provides the maximum of the exponential match, but now between 4 different dates. This is necessary if we want to compare matches
    across several columns and records.
    :param date_0:
    :param data_1:
    :param date_2:
    :param date_3:
    :param scale:
    :return:
    """
    return np.max([exponential_time_match(date_0, data_1, scale),
                   exponential_time_match(date_2, date_3, scale)])


def get_symmetric_max_exponential_time_matcher_with_scale(scale):
    """
    returns an `symmetric_max_exponential_time_match` function with a given scale. Useful for functional programming where we want to pre-specify the scale
    :param scale:
    :return:
    """
    return partial(symmetric_max_exponential_time_match, scale=scale)


def symmetric_min_exponential_time_match(date_0, data_1, date_2, date_3, scale):
    """
    See `symmetric_max_exponential_time_match` but now with min instead of max
    :param date_0:
    :param data_1:
    :param date_2:
    :param date_3:
    :param scale:
    :return:
    """
    return np.min([exponential_time_match(date_0, data_1, scale),
                   exponential_time_match(date_2, date_3, scale)])


def get_symmetric_min_exponential_time_matcher_with_scale(scale):
    """
    See `get_symmetric_max_exponential_time_matcher_with_scale` but now with min instead of max
    :param scale:
    :return:
    """
    return partial(symmetric_min_exponential_time_match, scale=scale)


class Distinctor(object):
    """
    Tool for calculating a match vector between two fuzzy entities that can be used for deduping multiple records.
    Usage example:

    >>> attribute_to_matcher_dict = {
        'attr_1': distinctor.inv_contrast_modulation,
        ('attr_2', attr_3'): get_symmetric_min_exponential_time_matcher_with_scale(10),
        'attr_4': fuzzy_ratio
        }

    >>> dstnctr = Distinctor(attribute_to_matcher_dict)
    >>> dstnctr.compute_match_vector(record_0, record_1)

    This returns a match vector based on the 4 attributes in the provided dictionary.
    """

    def __init__(self, attribute_to_matcher_dict):
        self.single_attribute_matchers = self._get_single_attribute_matchers(attribute_to_matcher_dict)
        self.two_attribute_matchers = self._get_two_attribute_matchers(attribute_to_matcher_dict)

    @staticmethod
    def _get_single_attribute_matchers(attribute_to_matcher_dict):
        return {attr: matcher for attr, matcher in attribute_to_matcher_dict.iteritems() if isinstance(attr, str)}

    @staticmethod
    def _get_two_attribute_matchers(attribute_to_matcher_dict):
        return {attr: matcher for attr, matcher in attribute_to_matcher_dict.iteritems() if (isinstance(attr, tuple) and (len(attr) == 2))}

    def compute_match_vector(self, record_0, record_1):
        dct = dict()
        dct.update(self._compute_one_attribute_matches(record_0, record_1))
        dct.update(self._compute_two_attribute_matches(record_0, record_1))
        return dct

    def _compute_one_attribute_matches(self, record_0, record_1):
        return {attr: matcher(record_0[attr], record_1[attr])
                for attr, matcher in self.single_attribute_matchers.iteritems()}

    def _compute_two_attribute_matches(self, record_0, record_1):
        return {attr[0] + '__to__' + attr[1]: matcher(record_0[attr[0]], record_1[attr[1]],
                                                      record_0[attr[1]], record_1[attr[0]])
                for attr, matcher in self.two_attribute_matchers.iteritems()}

    def compute_match_vector_mean(self, record_0, record_1):
        return np.mean([matcher(record_0[attr], record_1[attr]) for attr, matcher in self.single_attribute_matchers.iteritems()] +
                       [matcher(record_0[attr[0]], record_1[attr[1]],record_0[attr[1]], record_1[attr[0]]) for attr, matcher in self.two_attribute_matchers.iteritems()])
