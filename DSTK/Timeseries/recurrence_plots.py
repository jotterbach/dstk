import _recurrence_map


# def poincare_map(timeseries, threshold=0.1):
#     rec_map = _recurrence_map.recurrence_map(timeseries, timeseries)
#
#     return (rec_map < threshold).astype(int)

def poincare_map(ts, ts2=None, threshold=0.1):
    if ts2 is None:
        rec_map = _recurrence_map.recurrence_map(ts, ts)
    else:
        rec_map = _recurrence_map.recurrence_map(ts, ts2)

    return (rec_map < threshold).astype(int)
