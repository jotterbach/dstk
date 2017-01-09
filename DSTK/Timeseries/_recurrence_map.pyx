import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport fabs
from cython.parallel import prange, parallel


@cython.boundscheck(False)
def recurrence_map(np.ndarray[double, ndim=1, mode="c"] ts1, np.ndarray[double, ndim=1, mode="c"] ts2):

    cdef int n, m, i, j;
    n = ts1.shape[0]
    m = ts2.shape[0]

    cdef double[:, :] recMap = <double[:n, :n]>malloc(n * m * sizeof(double));

    with nogil, parallel():
        for i in prange(n, schedule='dynamic'):
            for j in range(m):
                recMap[i][j] = fabs(ts1[i] - ts2[j])

    return np.asarray(recMap)
