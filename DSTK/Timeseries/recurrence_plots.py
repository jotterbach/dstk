import _recurrence_map
import numpy as np


def poincare_map(ts, ts2=None, threshold=0.1):
    rec_dist = poincare_recurrence_dist(ts, ts2)
    return (rec_dist < threshold).astype(int)


def poincare_recurrence_dist(ts, ts2=None):
    if ts2 is None:
        return _recurrence_map.recurrence_map(ts, ts)
    else:
        return _recurrence_map.recurrence_map(ts, ts2)

