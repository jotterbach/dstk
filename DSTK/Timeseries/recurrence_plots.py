import _recurrence_map


def poincare_map(timeseries, threshold=0.1):
    rec_map = _recurrence_map.recurrence_map(timeseries)

    return (rec_map < threshold).astype(int)
