##########################################
# File: util.py                          #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################
# Imports
import re

# raise_if_not_shape
def raise_if_not_shape(name, A, shape):
    """Raise a `ValueError` if the np.ndarray `A` does not have dimensions
    `shape`."""
    if A.shape != shape:
        raise ValueError('{}.shape != {}'.format(name, shape))


# previous_float
PARSE_FLOAT_RE = re.compile(r'([+-]*)0x1\.([\da-f]{13})p(.*)')
def previous_float(x):
    """Return the next closest float (towards zero)."""
    s, f, e = PARSE_FLOAT_RE.match(float(x).hex().lower()).groups()
    f, e = int(f, 16), int(e)
    if f > 0:
        f -= 1
    else:
        f = int('f' * 13, 16)
        e -= 1
    return float.fromhex('{}0x1.{:013x}p{:d}'.format(s, f, e))

##############################################################################################
############################# GET BSPLINE FROM PREPROCESSED IMAGE ############################
##############################################################################################
# Parameterize the image based on final line drawing and tangent flow
# may need to cython the following code
def getParamter(img,etf):
    row_, col_ = img.shape
    mask = np.zeros((row_,col_))
    par_set = []
    for r in np.arange(row_):
        for c in np.arange(col_):
            if img[r,c]==0 and mask[r,c]==0:
                cos_theta = etf[r,c,0]
                sin_theta = etf[r,c,1]
                r_offset = int(np.round(sin_theta))
                c_offset = int(np.round(cos_theta))
                rp = r+r_offset
                cp = c+c_offset
                rn = r-r_offset
                cn = c-c_offset
                par = np.empty((0, 2)) # store coordinates for the current curve
                par = np.concatenate((par, [[r,c]]),axis=0)
                width = [] # store gradient-wise width of the current curve
                mask[r,c] = 1

                # get width in the gradient direction
                w, mask = getWidth(img,etf,mask,r,c)
                width.append(w)

                while (img[rp,cp]==0 and mask[rp,cp]==0) or (img[rn,cn]==0 and mask[rn,cn]==0):

                    # trace the current curve
                    if img[rp,cp]==0 and mask[rp,cp]==0:
                        par = np.concatenate((par, [[rp, cp]]), axis=0)
                        mask[rp,cp] = 1
                        w, mask = getWidth(img,etf,mask,rp,cp)
                        width.append(w)
                        cos_theta_p = etf[rp,cp,0]
                        sin_theta_p = etf[rp,cp,1]
                        r_offset_p = int(np.round(sin_theta_p))
                        c_offset_p = int(np.round(cos_theta_p))
                        rp += r_offset_p
                        cp += c_offset_p

                    if img[rn,cn]==0 and mask[rn,cn]==0:
                        par = np.concatenate((par, [[rn, cn]]), axis=0)
                        mask[rn,cn] = 1
                        w, mask = getWidth(img,etf,mask,rn,cn)
                        width.append(w)
                        cos_theta_n = etf[rn,cn,0]
                        sin_theta_n = etf[rn,cn,1]
                        r_offset_n = int(np.round(sin_theta_n))
                        c_offset_n = int(np.round(cos_theta_n))
                        rn -= r_offset_n
                        cn -= c_offset_n

                    wait = 1

                if par.shape[0]>2:
                    # c,u0,X = fit_bspline(par[:,0], par[:,1])
                    # par_set.append((c,u0,X,np.array(width)))
                    par_set.append(par)

    return par_set

# get curve width for a given pixel in the image
def getWidth(img,etf,mask,r,c):
    w = 1
    if img[r,c]== 0:
        g_cos_theta = etf[r,c,1]
        g_sin_theta = -etf[r,c,0]
        g_r_offset = int(np.round(g_sin_theta))
        g_c_offset = int(np.round(g_cos_theta))
        rgp = r+g_r_offset
        cgp = c+g_c_offset
        rgn = r-g_r_offset
        cgn = c-g_c_offset
        relevance_p = ((1-img[rgp,cgp])*(np.sum(etf[r,c]*etf[rgp,cgp],axis=0))>0.8)&(mask[rgp,cgp]==0)
        relevance_n = ((1-img[rgn,cgn])*(np.sum(etf[r,c]*etf[rgn,cgn],axis=0))>0.8)&(mask[rgn,cgn]==0)
        while relevance_p or relevance_n:
            if relevance_p:
                w += 1
                mask[rgp,cgp] = 1
                rgp += g_r_offset
                cgp += g_c_offset
            if relevance_n:
                w += 1
                mask[rgn,cgn] = 1
                rgn -= g_r_offset
                cgn -= g_c_offset
            relevance_p = ((1-img[rgp,cgp])*(np.sum(etf[r,c]*etf[rgp,cgp],axis=0))>0.8)&(mask[rgp,cgp]==0)
            relevance_n = ((1-img[rgn,cgn])*(np.sum(etf[r,c]*etf[rgn,cgn],axis=0))>0.8)&(mask[rgn,cgn]==0)
    return w, mask


# Redraw the picture from control points
def drawFromParameter(par_set,row,col):
    img = np.ones((row,col))
    for id, par in enumerate(par_set):
        for i in np.arange(par.shape[0]):
            img[par[i,0],par[i,1]] = 1.0*id/par_set.__len__()
    return img

############################## TEST CLUSTERING ##########################################
def getCluster(img,etf):
    ID = np.array(np.where(img==0))
    T = etf[img==0]
    # X = np.concatenate((ID.T,np.arctan2(T[:,0],T[:,1])[np.newaxis].T),axis=1)
    X = np.concatenate((ID.T,T),axis=1)
    X = StandardScaler().fit_transform(X)

    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    ##############################################################################
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    plt.subplot(1,2,1)
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    # show tangent flow by angle
    plt.subplot(1,2,2)
    for xyt in X:
        plt.plot(xyt[1],xyt[0],'o',markerfacecolor=str(xyt[2]/np.pi/2+0.5),
                 markeredgecolor='k', markersize=14)

    plt.show()
##############################################################################################
############################# GET BSPLINE FROM PREPROCESSED IMAGE ############################
##############################################################################################



##############################################################################################
######################################## TEST XDOG ###########################################
##############################################################################################
## code from: http://www.graco.c.u-tokyo.ac.jp/~tody/blog/2015/01/27/XDoG/
## Sharp image from scaled DoG signal.
#  @param  img        input gray image.
#  @param  sigma      sigma for small Gaussian filter.
#  @param  k_sigma    large/small sigma (Gaussian filter).
#  @param  p          scale parameter for DoG signal to make sharp.
def sharpImage(img, sigma, k_sigma, p):
    sigma_large = sigma * k_sigma
    G_small = cv2.GaussianBlur(img, (0, 0), sigma)
    G_large = cv2.GaussianBlur(img, (0, 0), sigma_large)
    S = G_small - p * G_large
    return S


## Soft threshold function to make ink rendering style.
#  @param  img        input gray image.
#  @param  epsilon    threshold value between dark and bright.
#  @param  phi        soft thresholding parameter.
def softThreshold(SI, epsilon, phi):
    T = np.zeros(SI.shape)
    SI_bright = SI >= epsilon
    SI_dark = SI < epsilon
    T[SI_bright] = 1.0
    T[SI_dark] = 1.0 + np.tanh(phi * (SI[SI_dark] - epsilon))
    return T


## XDoG filter.
#  @param  img        input gray image.
#  @param  sigma      sigma for sharpImage.
#  @param  k_sigma    large/small sigma for sharpImage.
#  @param  p          scale parameter for sharpImage.
#  @param  epsilon    threshold value for softThreshold.
#  @param  phi        soft thresholding parameter for softThreshold.
def XDoG(img, sigma, k_sigma, p, epsilon, phi):
    S = sharpImage(img, sigma, k_sigma, p)
    SI = np.multiply(img, S)
    T = softThreshold(SI, epsilon, phi)
    return T
##############################################################################################
######################################## TEST XDOG ###########################################
##############################################################################################

##############################################################################################
######################################## MISC ################################################
##############################################################################################
# get tangent from gradient
def tangent(g, gnorm):
    gradx = g[:, :, 0]
    grady = g[:, :, 1]
    theta = np.arctan2(grady, gradx)
    # tangent direction is counter-clock wise 90 deg from gradient direction
    beta = theta + np.pi / 2
    tanx = np.cos(beta)
    tany = np.sin(beta)

    # if gradient is zero, set tangent back to zero
    tanx[np.where(gnorm == 0)] = 0
    tany[np.where(gnorm == 0)] = 0

    # # recover the original gradient norm
    # tanx = tanx*gnorm
    # tany = tany*gnorm

    return np.dstack((tanx, tany))

# update gradient
def gradient(t, gnorm):
    tanx = t[:, :, 0]
    tany = t[:, :, 1]
    theta = np.arctan2(tany, tanx)
    # tangent direction is counter-clock wise 90 deg from gradient direction
    beta = theta - np.pi / 2
    gradx = np.cos(beta)
    grady = np.sin(beta)

    # if gradient is zero, set tangent back to zero
    gradx[np.where(gnorm == 0)] = 0
    grady[np.where(gnorm == 0)] = 0

    # recover the original gradient norm
    gradx = gradx*gnorm
    grady = grady*gnorm

    return np.dstack((gradx, grady))

# define neighbours
def get_neighbours(r):
    neighbour = np.empty((0, 2))
    for i in np.arange(-r, r + 1):
        rr = np.round(np.sqrt(r ** 2 - i ** 2))
        for j in np.arange(-rr, rr + 1):
            neighbour = np.concatenate((neighbour, [[i, j]]), axis=0)
    return neighbour.astype(np.int64)
##############################################################################################
######################################## MISC ################################################
##############################################################################################