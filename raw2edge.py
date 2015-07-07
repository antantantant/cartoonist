__author__ = 'yiren'
import os.path
import cv2
import numpy as np
from matplotlib import pyplot as plt

## TEST XDOG
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
    S = (1 + p) * G_small - p * G_large
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


# get normalized gradient from sobel filter
def sobel(img):
    gradx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    grady = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    grad = np.dstack((gradx, grady))
    grad_norm = np.linalg.norm(grad, axis=2)
    return [grad, grad_norm]


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


# main function to update tangent
# ita, r: wm parameter, neighbourhood radius
def edge_tangent_flow(img, ETF_counter, ita, r):
    (grad, grad_norm) = sobel(img)
    tan = tangent(grad, grad_norm)
    original_tan = tan.copy()
    (xmax, ymax) = img.shape

    # calculate weights in neighbourhood and filter out zero tangent pixels
    # nonzero_tan_id_tuple = ((tan[:, :, 0] != 0) + (tan[:, :, 1] != 0) + 0 > 0).nonzero()
    nonzero_tan_id_tuple = (tan[:,:,0] != np.nan).nonzero()
    nonzero_tan_id = np.dstack((nonzero_tan_id_tuple[0], nonzero_tan_id_tuple[1]))[0]
    # w = np.empty((0,7)) # xid (2 col), yid (2 col), wd, wm, phi
    tan_new = np.zeros(tan.shape)
    neighbour_pattern = get_neighbours(r)
    for ETF_loop in np.arange(ETF_counter):
        for id, xid in enumerate(nonzero_tan_id):
            if np.mod(id, nonzero_tan_id.size/2/10) == 0:
                print(ETF_loop.astype(str) + '==' + np.round(id * 2.0 / nonzero_tan_id.size * 100).astype(str) + '%')
            # neighbour_set = np.append(neighbour_set, [[np.concatenate((xid, neighbours), axis=0)]])
            yid = xid + neighbour_pattern
            yid = yid[(yid[:, 0] >= 0) * (yid[:, 0] < xmax) * (yid[:, 1] >= 0) * (yid[:, 1] < ymax)]
            yid = yid[(grad_norm[yid[:, 0], yid[:, 1]] != 0)]
            wd = np.maximum(np.sum(tan[xid[0], xid[1], :] * tan[yid[:, 0], yid[:, 1], :], axis=1), 0.01)
            wm = 0.5 * (1 + np.tanh(ita * (grad_norm[xid[0], xid[1]] - grad_norm[yid[:, 0], yid[:, 1]])))
            # phi = (np.sum(tan[xid[0], xid[1], :] * tan[yid[:, 0], yid[:, 1], :], axis=1) > 0) * 2 - 1
            # w = np.concatenate((w, [[xid[0], xid[1], yid[0], yid[1], wd, wm, phi]]), axis=0)
            t = 0.5*tan[xid[0],xid[1],:] # ws=1, wm=0.5, wd=1
            t = t + np.sum((wd * wm)[np.newaxis].T * tan[yid[:, 0], yid[:, 1], :], axis=0)

            tan_new[xid[0], xid[1]] = t / (np.linalg.norm(t))
        # update tangent
        tan = tan_new.copy()
        tan_new = np.zeros(tan.shape)
        # update gradient
        grad = gradient(tan, grad_norm)


    return grad, grad_norm, original_tan, tan

## main code
address = 'raw_image/'
filename = 'test2.jpg'
img = cv2.imread(address + filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## shrink the image to a pre-defined standard
gray = cv2.resize(gray, None, fx=0.50, fy=0.50, )
plt.subplot(1, 4, 1)
plt.imshow(gray, 'gray')

# ## XDoG filter to get binary image
# filteredimg_address = address + 'filtered_' + filename + '.out'
# if os.path.isfile(filteredimg_address):
#     filteredimg = np.loadtxt(filteredimg_address)
# else:
#     sigma = 1.0
#     k_sigma = 1.6
#     p = 100.0
#     epsilon = 0.5
#     phi = 1.0
#     filteredimg = XDoG(gray, sigma, k_sigma, p, epsilon, phi)
#     np.savetxt(filteredimg_address, filteredimg)
# plt.subplot(1,4,2)
# plt.imshow(filteredimg, 'gray')

## Calculate tangent flow
ETF_counter = 3
ita = 1
r = 5
grad, grad_norm, tan, tan_ETF = edge_tangent_flow(gray, ETF_counter, ita, r)
tanimg_address = address + 'tan_' + filename + '.out'
np.save(tanimg_address, tan)
ETFimg_address = address + 'ETF_' + str(ETF_counter) + filename + '.out'
np.save(ETFimg_address, tan_ETF)
plt.subplot(1, 4, 3)
plt.imshow(tan[:, :, 0], 'gray')
plt.subplot(1, 4, 4)
plt.imshow(tan_ETF[:, :, 0], 'gray')
plt.show()
