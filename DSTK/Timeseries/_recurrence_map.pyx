import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport fabs
from cython.parallel import prange, parallel


@cython.boundscheck(False)
def recurrence_map(np.ndarray[double, ndim=1, mode="c"] ts1,
                   np.ndarray[double, ndim=1, mode="c"] ts2):

    cdef int n, m, i, j;
    n = ts1.shape[0]
    m = ts2.shape[0]

    cdef double[:, :] recMap = <double[:n, :m]>malloc(n * m * sizeof(double));
    cdef np.ndarray out;

    try:
        with nogil, parallel():
            for i in prange(n, schedule='dynamic'):
                for j in range(m):
                    recMap[i][j] = fabs(ts1[i] - ts2[j])

        # The following copy is needed to properly GC the objects.
        # It impacts performance but I haven't figured out a way how to
        # turn a memory view into a NumPy array.
        out = np.copy(np.asarray(recMap))
        return out
    finally:
        free(&recMap[0, 0])
