__author__ = 'yiren'
import os.path
import cv2
import numpy as np
from matplotlib import pyplot as plt
import lic_internal
import FDoGedge

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


# get normalized gradient from sobel filter
def sobel(img):
    gradx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    grady = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    grad = np.dstack((gradx, grady))
    grad_norm = np.linalg.norm(grad, axis=2)
    st = np.dstack((gradx*gradx, grady*grady, gradx*grady))
    return grad, grad_norm, st


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
def edge_tangent_flow(img):
    # step 1: Calculate the struture tensor
    grad, grad_norm, st = sobel(img)
    row_, col_ = img.shape

    # step 2: Gaussian blur the struct tensor. sst_sigma = 2.0
    sigma_sst = 2.0
    gaussian_size = int((sigma_sst*2)*2+1)
    blur = cv2.GaussianBlur(st, (gaussian_size,gaussian_size), sigma_sst)

    tan_ETF = np.zeros((row_,col_,2))
    E = blur[:,:,0]
    G = blur[:,:,1]
    F = blur[:,:,2]

    # plt.subplot(2,3,1)
    # plt.imshow(E, 'gray')
    # plt.subplot(2,3,2)
    # plt.imshow(G, 'gray')
    # plt.subplot(2,3,3)
    # plt.imshow(F, 'gray')

    lambda2 = 0.5*(E+G-np.sqrt((G-E)*(G-E)+4.0*F*F))
    v2x = (lambda2 - G != 0) * (lambda2 - G) + (lambda2 - G == 0) * F
    v2y = (lambda2 - G != 0) * F + (lambda2 - G == 0) * (lambda2 -E)

    # plt.subplot(2,3,4)
    # plt.imshow(v2x, 'gray')
    # plt.subplot(2,3,5)
    # plt.imshow(v2y, 'gray')

    v2x = cv2.GaussianBlur(v2x, (gaussian_size,gaussian_size), sigma_sst)
    v2y = cv2.GaussianBlur(v2y, (gaussian_size,gaussian_size), sigma_sst)

    v2 = np.sqrt(v2x*v2x+v2y*v2y)
    tan_ETF[:,:,0] = v2x/(v2+0.0000001)*((v2!=0)+0)
    tan_ETF[:,:,1] = v2y/(v2+0.0000001)*((v2!=0)+0)

    return tan_ETF

# Visualize a vector field by using LIC (Linear Integral Convolution).
def visualizeByLIC(vf):
    row_,col_,dep_ = vf.shape
    texture = np.random.rand(col_,row_).astype(np.float32)
    kernellen=9
    kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)
    kernel = kernel.astype(np.float32)
    vf = vf.astype(np.float32)
    img = lic_internal.line_integral_convolution(vf, texture, kernel)
    return img

    # dpi = 100
    # plt.bone()
    # plt.clf()
    # plt.axis('off')
    # plt.figimage(img)
    # plt.gcf().set_size_inches((col_/float(dpi),row_/float(dpi)))
    # plt.savefig("flow-image.png",dpi=dpi)


# Visualize a vector field by arrow
def visualizeByArrow(vf,img):
    row_, col_, dep_ = vf.shape
    dx = vf[:,:,0]
    dy = vf[:,:,1]
    mag = vf[:,:,2]
    angle = np.arctan2(dy,dx)
    angle[mag<=0]=0
    angle[(mag>0)*(np.abs(dx)<0.0000001)] = 0.5*np.pi
    c = np.ones((row_,1))*np.arange(col_)
    r = np.arange(row_)[np.newaxis].T*np.ones((1,col_))
    p = np.dstack((c,r)).astype(int)
    q = np.dstack((c-np.round(0.5*mag*np.cos(angle)).astype(int),r-np.round(0.5*mag*np.sin(angle)))).astype(int)

    for r_ in np.arange(0,row_,20):
        for c_ in np.arange(0,col_,20):
            img = cv2.line(img,(p[r_,c_,0],p[r_,c_,1]),(q[r_,c_,0],q[r_,c_,1]),(0,0,0),1)
    plt.imshow(img)
    plt.show()



## main code
address = 'raw_image/'
filename = 'ghibli1.jpg'
img = cv2.imread(address + filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
row_, col_ = gray.shape

## shrink the image to a pre-defined standard
gray = cv2.resize(gray, None, fx=0.8, fy=0.8, )
plt.subplot(2,3,1)
plt.imshow(gray, 'gray')

# ## XDoG filter to get binary image
# sigma = 1.0
# k_sigma = 1.6
# p = 0.99
# epsilon = 0.005
# phi = 1.0
# # filteredimg_address = address + 'filtered_' + filename + \
# #                       '_sigma_' + str(sigma) + '_p_' + str(p) + '.out'
# # if os.path.isfile(filteredimg_address):
# #     filteredimg = np.loadtxt(filteredimg_address)
# # else:
# filteredimg = XDoG(gray, sigma, k_sigma, p, epsilon, phi)
# #     np.savetxt(filteredimg_address, filteredimg)
# plt.subplot(2,3,2)
# plt.imshow(filteredimg, 'gray')
#
# gray_ = gray.copy()
# gray_[filteredimg<1.0]=0.0
#
# plt.subplot(2,3,3)
# plt.imshow(gray_, 'gray')

gray_ = gray.copy()

## Calculate tangent flow
tan_ETF = edge_tangent_flow(gray_)
lic_img = visualizeByLIC(tan_ETF)
plt.subplot(2,3,2)
plt.imshow(lic_img, 'gray')

## FDoG loop
fdog_loop = 3
for count in np.arange(fdog_loop):
    ## Get FDoG Edge
    sigma_e = 1.0
    sigma_r = 1.6
    sigma_m = 3.0
    tau = 0.99
    phi = 2.0
    threshold = 0.2
    fdog_img, f0, f1 = FDoGedge.getFDoGedge(tan_ETF.astype(np.float64),gray_.astype(np.float64),
                                    sigma_e,sigma_r,sigma_m,tau,phi,threshold)
    plt.subplot(2,3,count+4)
    plt.imshow(fdog_img,'gray')

    # plt.subplot(2,3,5)
    # plt.imshow(f0,'gray')
    #
    # plt.subplot(2,3,6)
    # plt.imshow(f1,'gray')

    gray_[fdog_img<255]=0.0

    # filteredimg = XDoG(gray_, sigma, k_sigma, p, epsilon, phi)
    # plt.subplot(2,3,count+4)
    # plt.imshow(filteredimg,'gray')
    # gray_ = gray.copy()
    # gray_[filteredimg<1.0]=0.0

plt.show()
# dpi = 100
# plt.bone()
# plt.clf()
# plt.axis('off')
# plt.figimage(fdog_img)
# plt.gcf().set_size_inches((col_/float(dpi),row_/float(dpi)))
# plt.savefig("fdogedge-image.png",dpi=dpi)
