import _recurrence_map


def poincare_map(timeseries, threshold=0.1, num_threads=None):
    rec_map = _recurrence_map.recurrence_map(timeseries, num_threads=num_threads)

    return (rec_map < threshold).astype(int)
