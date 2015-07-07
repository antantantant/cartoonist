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

# Get FDoG edge
def getFDoGedge(np.ndarray[double, ndim=3] etf, np.ndarray[double, ndim=2] img,
                double sigma_e, double sigma_r, double sigma_m, double tau, double phi, double threshold):

    cdef int row_, col_
    cdef int neighbour1, neighbour2, c_offset, r_offset
    cdef double sin_theta, cos_theta, sum_diff, sum_1
    cdef np.ndarray[double, ndim=2] fdog_img, f0, f1, u1
    cdef np.ndarray[double, ndim=1] diff_gaussian_weights, gaussian_weights
    cdef np.ndarray[double, ndim=1] sample_pixels1, sample_pixels2

    row_ = img.shape[0]
    col_ = img.shape[1]

    fdog_img = np.zeros((row_,col_),dtype=np.float64)
    f0 = np.ones((row_,col_))
    f1 = np.ones((row_,col_))
    u1 = np.zeros((row_,col_))

    neighbour1 = int(np.ceil(phi*sigma_r))
    diff_gaussian_weights = _diffgaussianweights(neighbour1, sigma_e, sigma_r, tau)
    neighbour2 = np.ceil(phi*sigma_m)
    gaussian_weights = _gaussianweights(neighbour2, sigma_m)

    # Step 1: do DoG along the gradient direction.
    for r in np.arange(neighbour1,row_-neighbour1):
        for c in np.arange(neighbour1,col_-neighbour1):
            cos_theta = etf[r,c,1]
            sin_theta = -etf[r,c,0]
            sample_pixels1 = np.zeros((neighbour2*2+1,),dtype=np.float64)
            sample_pixels1[neighbour1] = img[r,c]
            for k in np.arange(1,neighbour1+1):
                r_offset = int(np.round(sin_theta*k))
                c_offset = int(np.round(cos_theta*k))
                sample_pixels1[neighbour1+k] = img[r+r_offset,c+c_offset]
                sample_pixels1[neighbour1-k] = img[r-r_offset,c-c_offset]
            sum_diff = 0.0
            for k in np.arange(2*neighbour1+1):
                sum_diff += sample_pixels1[k]*diff_gaussian_weights[k]
            f0[r,c] = sum_diff

    # Step 2: do Gaussian blur along tangent direction.
    for r in np.arange(row_):
        for c in np.arange(col_):
            sample_pixels2 = np.zeros((neighbour2*2+1,),dtype=np.float64)
            sample_pixels2[neighbour2] = f0[r,c]

            cos_theta = etf[r,c,0]
            sin_theta = etf[r,c,1]
            r_offset = 0
            c_offset = 0
            for k in np.arange(neighbour2):
                r_offset += int(np.round(sin_theta))
                c_offset += int(np.round(cos_theta))
                if (r+r_offset<row_) and (c+c_offset<col_) and (r+r_offset>=0) and (c+c_offset>=0):
                    sample_pixels2[neighbour2+k+1] = f0[r+r_offset,c+c_offset]
                    cos_theta = etf[r+r_offset,c+c_offset,0]
                    sin_theta = etf[r+r_offset,c+c_offset,1]
                else:
                    sample_pixels2[neighbour2+k+1] = 0
                    continue

            cos_theta = etf[r,c,0]
            sin_theta = etf[r,c,1]
            r_offset = 0
            c_offset = 0
            for k in np.arange(neighbour2):
                r_offset += int(np.round(sin_theta))
                c_offset += int(np.round(cos_theta))
                if (r-r_offset>=0) and (c-c_offset>=0) and (r-r_offset<row_) and (c-c_offset<col_):
                    sample_pixels2[neighbour2-k-1] = f0[r-r_offset,c-c_offset]
                    cos_theta = etf[r-r_offset,c-c_offset,0]
                    sin_theta = etf[r-r_offset,c-c_offset,1]
                else:
                    sample_pixels2[neighbour2-k-1] = 0
                    continue

            sum_1 = 0.0
            for k in np.arange(2*neighbour2+1):
                sum_1 += sample_pixels2[k]*gaussian_weights[k]
            f1[r,c] = sum_1

            if np.tanh(f1[r,c])>threshold-1:
                u1[r,c] = 0
                fdog_img[r,c] = 255
            else:
                u1[r,c] = 255
                fdog_img[r,c] = 0

    return fdog_img, f0, f1
    
