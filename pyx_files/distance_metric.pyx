import numpy as np
cimport numpy as np

# Define a distance metric for sketches
# x[0]: x-coordinate
# x[1]: y-coordinate
# x[2]: tangent flow angle [-1,1]
cdef double _metric(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] y, int c):
    cdef double d, dd
    cdef np.ndarray[double, ndim=1] dx
    dd = np.sqrt((x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]))
    if c==1 and dd<0.5:
        d = 0
    else:
        d = dd
    d += 1-np.abs(x[2]*y[2]+x[3]*y[3])
    dx = x[0:2]-y[0:2]
    dx /= np.linalg.norm(dx)
    d += (1-np.abs(np.sum(dx*x[2:4])))*dd
    d += (1-np.abs(np.sum(dx*y[2:4])))*dd
    return d

def distance(np.ndarray[double, ndim=2] X, np.ndarray[int, ndim=2] C):
    cdef np.ndarray[double, ndim=2] D
    cdef int n, i, j

    n = X.shape[0]
    D = np.zeros((n,n))
    for i in np.arange(n):
        for j in np.arange(i+1,n):
            D[i,j] = _metric(X[i,:],X[j,:],C[i,j])
            D[j,i] = D[i,j]
    return D
