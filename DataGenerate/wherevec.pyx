cimport numpy as np
cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def wherevec_cython(long[:] vec, long[:, :] matrix):
    cdef Py_ssize_t i, j, nrows, ncols
    cdef long[:] row_int

    nrows = matrix.shape[0]
    ncols = matrix.shape[1]

    for i in range(nrows):
        row_int = matrix[i]
        for j in range(ncols):
            if row_int[j] != vec[j]:
                break
        else:
            return i
    return -1
