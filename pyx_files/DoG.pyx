import numpy as np
cimport numpy as np

# get Diff Gaussian Weights
cdef np.ndarray[double, ndim=1] _diffgaussianweights(int neighbour, double sigma_e, double sigma_r, double tau):
    cdef np.ndarray[double, ndim=1] gaussian_e, gaussian_r, weights
    cdef double sum

    gaussian_e = _gaussianweights(neighbour, sigma_e)
    gaussian_r = _gaussianweights(neighbour, sigma_r)
    weights = np.zeros((neighbour*2+1,),dtype=np.float64)
    for i in np.arange(neighbour*2+1):
        weights[i] = gaussian_e[i]-tau*gaussian_r[i]

    return weights

# get Gaussian Weights
cdef np.ndarray[double, ndim=1] _gaussianweights(int neighbour, double sigma):
    cdef np.ndarray[double, ndim=1] weights, gaussian
    cdef double term1, term2, sum

    weights = np.zeros((neighbour*2+1,),dtype=np.float64)
    gaussian = np.zeros((neighbour*2+1,),dtype=np.float64)
    term1 = 1.0/(np.sqrt(2*np.pi)*sigma)
    term2 = -1.0/(2*(sigma**2))
    weights[neighbour] = term1
    sum = term1
    for i in np.arange(1,neighbour+1):
        weights[neighbour+i] = np.exp((i**2)*term2)*term1
        weights[neighbour-i] = weights[neighbour+i]
        sum+=weights[neighbour+i]+weights[neighbour-i]
    for i in np.arange(neighbour*2+1):
        weights[i] = weights[i]/sum
    return weights

# def double DoG(np.ndarray[double, ndim=2] img, np.ndarray[double, ndim=1] x, double sigma_e,double sigma_r, int phi):
#     cdef r, dog
#
#     r =