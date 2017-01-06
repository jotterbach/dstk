import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport fabs
from cython.parallel import prange, parallel

@cython.boundscheck(False)
def recurrence_map(np.ndarray[double, ndim=1, mode="c"] ts):

    cdef int n, i, j;
    n = ts.shape[0]

    cdef double[:, :] recMap = <double[:n, :n]>malloc((n ** 2) * sizeof(double));
    cdef double dist_val;

    with nogil, parallel(num_threads=None):
        for i in prange(n, schedule='dynamic'):
            for j in range(i, n):
                if i == j:
                    recMap[i][j] = 0.0
                else:
                    dist_val = fabs(ts[i] - ts[j])
                    recMap[i][j] = dist_val
                    recMap[j][i] = dist_val

    return np.asarray(recMap)
