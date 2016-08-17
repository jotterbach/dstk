import Distinctor.distinctor as dst
import pandas as pd
from datetime import date

data_dct ={
    'a': [1, 2, 3],
    'b': ['abc', 'bc', 'z'],
    'c': [1, 3, 4],
    'd': [date(2015, 1, 1), date(2015, 1, 2), date(2015, 1, 3)],
    'e': [date(2015, 1, 3), date(2015, 1, 4), date(2015, 1, 5)]
}

test_df = pd.DataFrame.from_dict(data_dct)

attribute_to_matcher_dct = {
    'a': dst.contrast_modulation,
    'b': dst.fuzzy_ratio,
    'c': dst.inv_luminance_ratio,
    ('d', 'e'): dst.get_symmetric_max_exponential_time_matcher_with_scale(1)
}

dstnctr = dst.Distinctor(attribute_to_matcher_dct)


def test_distinctor():

    assert dstnctr.compute_match_vector(test_df.ix[0], test_df.ix[1]) == {'a': 0.33333333333333331, 'c': 0.33333333333333331, 'b': 0.8, 'd__to__e': 0.36787944117144233}
    assert dstnctr.compute_match_vector(test_df.ix[0], test_df.ix[2]) == {'a': 0.50000000000000011, 'c': 0.25, 'b': 0.0, 'd__to__e': 1.0}
    assert dstnctr.compute_match_vector(test_df.ix[1], test_df.ix[2]) == {'a': 0.20000000000000004, 'c': 0.75, 'b': 0.0, 'd__to__e': 0.36787944117144233}


def test_distinctor_mean():

    assert dstnctr.compute_match_vector_mean(test_df.ix[0], test_df.ix[1]) == 0.45863652695952728
    assert dstnctr.compute_match_vector_mean(test_df.ix[0], test_df.ix[2]) == 0.4375
    assert dstnctr.compute_match_vector_mean(test_df.ix[1], test_df.ix[2]) == 0.32946986029286063

