__author__ = 'yiren'
import os.path
import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt

# for image preprocessing
import lic_internal
import FDoGedge

# for regression with bspline
from uniform_bspline import UniformBSpline
from fit_uniform_bspline import UniformBSplineLeastSquaresOptimiser
from scipy.spatial.distance import cdist

# for clustering
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import networkx as nx

# get normalized gradient from sobel filter
def sobel(img):
    gradx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    grady = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    grad = np.dstack((gradx, grady))
    grad_norm = np.linalg.norm(grad, axis=2)
    st = np.dstack((gradx*gradx, grady*grady, gradx*grady))
    return grad, grad_norm, st

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

    lambda2 = 0.5*(E+G-np.sqrt((G-E)*(G-E)+4.0*F*F))
    v2x = (lambda2 - G != 0) * (lambda2 - G) + (lambda2 - G == 0) * F
    v2y = (lambda2 - G != 0) * F + (lambda2 - G == 0) * (lambda2 -E)
    # v2x = cv2.GaussianBlur(v2x, (gaussian_size,gaussian_size), sigma_sst)
    # v2y = cv2.GaussianBlur(v2y, (gaussian_size,gaussian_size), sigma_sst)
    v2 = np.sqrt(v2x*v2x+v2y*v2y)
    tan_ETF[:,:,0] = v2x/(v2+0.0000001)*((v2!=0)+0)
    tan_ETF[:,:,1] = v2y/(v2+0.0000001)*((v2!=0)+0)

    # plt.subplot(1,3,1)
    # plt.imshow(tan_ETF[:,:,0],'gray')
    # plt.subplot(1,3,2)
    # plt.imshow(tan_ETF[:,:,1],'gray')
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

# fit bspline code from Mark Fuge
# https://github.com/IDEALLab/semantic-design-languages
def fit_bspline(x, y, dim = 2, degree=2, num_control_points = 10,
                is_closed = False, num_init_points=1000):
    ''' Fits and returns a bspline curve to the given x and y points

        Parameters
        ----------
        x : list
            data x-coordinates
        y : list
            data y-coordinates
        dim : int
            the dimensionality of the dataset (default: 2)
        degree : int
            the degree of the b-spline polynomial (default: 2)
        num_control_points : int
            the number of b-spline control points (default: 20)
        is_closed : boolean
            should the b-spline be closed? (default: false)
        num_init_points : int
            number of initial points to use in the b-spline parameterization
            when starting the regression. (default: 1000)

        Returns
        -------
        c: a UniformBSpline object containing the optimized b-spline
    '''
    # TODO: extract dimensionality from the x,y dataset itself
    num_data_points = len(x)
    c = UniformBSpline(degree, num_control_points, dim, is_closed=is_closed)
    Y = np.c_[x, y] # Data num_points by dimension
    # Now we need weights for all of the data points
    w = np.empty((num_data_points, dim), dtype=float)
    # Currently, every point is equally important
    w.fill(1) # Uniform weight to the different points
    # Initialize `X` so that the uniform B-spline linearly interpolates between
    # the first and last noise-free data points.
    t = np.linspace(0.0, 1.0, num_control_points)[:, np.newaxis]
    X = Y[0] * (1 - t) + Y[-1] * t
    # NOTE: Not entirely sure if the next three lines are necessary or not
    m0, m1 = c.M(c.uniform_parameterisation(2), X)
    x01 = 0.5 * (X[0] + X[-1])
    X = (np.linalg.norm(Y[0] - Y[-1]) / np.linalg.norm(m1 - m0)) * (X - x01) + x01
    # Regularization weight on the control point distance
    # This specifies a penalty on having the b-spline control points close
    # together, and in some sense prevents over-fitting. Change this is the
    # curve doesn't capture the curve variation well or smoothly enough
    lambda_ = 0.5
    # These parameters affect the regression solver.
    # Presently, they are disabled below, but you can think about enabling them
    # if that would be useful for your use case.
    max_num_iterations = 1000
    min_radius = 0
    max_radius = 400
    initial_radius = 100
    # Initialize U
    u0 = c.uniform_parameterisation(num_init_points)
    D = cdist(Y, c.M(u0, X))
    u = u0[D.argmin(axis=1)]
    # Run the solver
    (u, X, has_converged, states, num_iterations,
        time_taken) = UniformBSplineLeastSquaresOptimiser(c,'lm').minimise(
        Y, w, lambda_, u, X,
        #max_num_iterations = max_num_iterations,
        #min_radius = min_radius,
        #max_radius = max_radius,
        #initial_radius = initial_radius,
        return_all=True)
    return c,u0,X

def getCluster(img,etf):
    ID = np.array(np.where(img==0)).T
    T = etf[img==0]
    T = T[:,::-1] # the axes were FLIPPED for tangent flow calculation, now flip back!
    IDnorm = StandardScaler().fit_transform(ID.astype(float))
    X = np.concatenate((IDnorm,T),axis=1)
    # X = np.concatenate((IDnorm,np.arctan2(T[:,0],T[:,1])[np.newaxis].T),axis=1)
    # X = np.concatenate((ID.T,T,np.arctan2(T[:,0],T[:,1])[np.newaxis].T),axis=1)

    ## precompute connectivity
    connectivity_matix_address = 'connectivity.out'
    if os.path.isfile(connectivity_matix_address+'.npy'):
        C = np.load(connectivity_matix_address+'.npy')
    else:
        G = createGraph(ID,ID.shape[0])
        C = calConnectivity(G,ID.shape[0])
        np.save(connectivity_matix_address, C)

    ## precompute the distance matrix
    distance_matix_address = 'distance.out'
    if os.path.isfile(distance_matix_address+'.npy'):
        D = np.load(distance_matix_address+'.npy')
    else:
        D = distance(X,C.astype(int))
        np.save(distance_matix_address, D)

    # db = DBSCAN(metric=distance_metric.sketchmetric, eps=0.5, min_samples=10).fit(X)
    db = DBSCAN(metric='precomputed', eps=.1, min_samples=10).fit(D)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    ##############################################################################
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    plt.subplot(1,3,2)
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = ID[class_member_mask & core_samples_mask,:]
        ax1 = plt.gca()
        ax1.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = ID[class_member_mask & ~core_samples_mask,:]
        ax1.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)
        ax1.invert_yaxis()
    # # show tangent flow
    # plt.subplot(1,3,2)
    # for id, xyt in enumerate(X):
    #     plt.plot(ID[id,1],ID[id,0],'o',markerfacecolor=str(xyt[2]/2+0.5),
    #              markeredgecolor='k', markersize=14)
    # plt.subplot(1,3,3)
    # for id, xyt in enumerate(X):
    #     plt.plot(ID[id,1],ID[id,0],'o',markerfacecolor=str(xyt[3]/2+0.5),
    #              markeredgecolor='k', markersize=14)
    # plt.show()
    return ID, labels

# create graph from image
def createGraph(ID,N,etf):
    G = nx.Graph()
    G.add_nodes_from(np.arange(N))
    for n, id in enumerate(ID):
        for dx in [[0,1],[1,0],[1,1],[1,-1]]:
            idd = id.copy()
            idd[0] += dx[0]
            idd[1] += dx[1]
            n1 = np.where(np.logical_and(idd[1]==ID[:,1], idd[0]==ID[:,0]))[0]
            if n1.size == 1:
                G.add_weighted_edges_from([(n,n1[0],distance(n1[0],n,ID,etf))])
    return G

def distance(i,j,ID,etf):
    x1 = ID[i,:].astype(float)
    x2 = ID[j,:].astype(float)
    t1 = etf[x1[0],x1[1],:]
    t2 = etf[x2[0],x2[1],:]
    dx = (x1-x2)/np.linalg.norm(x1-x2)
    dx = dx[::-1] # the coordinate definition for ID (image) and etf are opposite
    d1 = np.array([1.0 - np.abs(np.dot(t1,t2)), 1.0 - np.abs(np.dot(dx,t1)), 1.0 - np.abs(np.dot(dx,t2))])
    d1[d1<0.1] = 0
    d1[d1>0.5] = 10
    return sum(d1)

def distance_all(X):
    n = X.shape[0]
    tangent = X[:,2:4]
    angle = 1.0-np.abs(np.dot(tangent,tangent.T))

    D = 1000000.0*np.ones((n,n))
    possibleID = np.array(np.where(np.logical_and(angle<0.3, C==1))).T.astype(int)
    d = angle[possibleID[:,0],possibleID[:,1]]
    X1 = X[possibleID[:,0],0:2]
    X2 = X[possibleID[:,1],0:2]
    dd = np.linalg.norm(X1-X2,axis=1)
    d[dd>=0.5] += dd[dd>=0.5]
    T1 = tangent[possibleID[:,0],:]
    T2 = tangent[possibleID[:,1],:]
    f1 = 10*(dd-np.abs(np.sum((X1-X2)*T1,axis=1)))
    f2 = 10*(dd-np.abs(np.sum((X1-X2)*T2,axis=1)))
    d += f1 + f2
    D[possibleID[:,0],possibleID[:,1]] = d
    return D

# calculate connectivity
def calConnectivity(G,N):
    # C = nx.all_pairs_node_connectivity(G)
    # CC = np.empty((0,N))
    # for c in C.values():
    #     CC = np.concatenate((CC, np.array(c.values())[np.newaxis]))
    connect_list = sorted(nx.connected_components(G), key = len, reverse=True)
    C = np.zeros((N,N))
    for list in connect_list:
        l = np.array(list)
        lr = np.kron(np.ones((1,l.size)),l)[0].astype(int)
        lc = np.kron(l,np.ones((1,l.size)))[0].astype(int)
        C[lr,lc] = 1
    np.fill_diagonal(C,0)
    # for i in np.arange(N):
    #     for j in np.arange(i+1,N):
    #         C[i,j] = nx.has_path(G,i,j) + 0.0
    #         C[j,i] = C[i,j]
    return connect_list, C

# find tip and fork points
def findTipPoints(ID, connected_group, row_, col_):
    tip_img = np.ones((row_,col_)) # create temp white image
    tip_points = []
    fork_points = []

    # plt.subplot(1,3,2)
    # ax2 = plt.gca()
    for group in connected_group:
        if group.__len__() >= 3: # ignore small groups
            xy = ID[group,:]
            temp_img = np.ones((row_,col_)) # create temp white image
            temp_img[xy[:,0],xy[:,1]] = 0 # set subgraph as a black pattern
            s = np.zeros((xy.shape[0],)) # initialize score
            for d in np.arange(3,5,1): # multi-scale filter
                for id, coord in enumerate(xy):
                    for r_ in np.arange(-d,d+1):
                        for c_ in np.arange(-d,d+1):
                            if r_!=0 or c_!=0:
                                if coord[0]+r_>=0 and coord[0]+r_<row_ and coord[1]+c_>=0 and coord[1]+c_<col_:
                                    s[id] -= temp_img[coord[0]+r_,coord[1]+c_] # more white areas means closer to tip
                                else:
                                    s[id] -= 1
            # temp_img[xy[:,0],xy[:,1]] = s
            # tip_img += temp_img

            s -= np.min(s) # set s to be non-negative
            s /= (np.max(s)+0.0000001) # set s to [0,1]

            # cluster s to get the tips and fork points
            db = KMeans(n_clusters=3).fit(s[np.newaxis].T)
            labels = db.labels_
            u_labels = set(labels)
            sum_s = np.zeros((3,))
            for id, l in enumerate(u_labels):
                sum_s[id] = np.average(s[labels==l])
            potential_fork = np.array(group)[labels==list(u_labels)[np.where(sum_s==np.max(sum_s))[0][0]]]
            potential_tip = np.array(group)[labels==list(u_labels)[np.where(sum_s==np.min(sum_s))[0][0]]]
            s_fork = s[labels==list(u_labels)[np.where(sum_s==np.max(sum_s))[0][0]]]
            s_tip = s[labels==list(u_labels)[np.where(sum_s==np.min(sum_s))[0][0]]]

            # for ipf, pf in enumerate(potential_fork):
            #     ax2.plot(pf[1], pf[0], 'o', markerfacecolor=str(s_fork[ipf]),
            #              markeredgecolor='k', markersize=10)
            # for ipt, pt in enumerate(potential_tip):
            #     ax2.plot(pt[1], pt[0], 'o', markerfacecolor=str(s_tip[ipt]),
            #              markeredgecolor='k', markersize=10)
            # ax2.invert_yaxis()

            # cluster tips and fork points and pick one out of each cluster
            picked_tip = []
            picked_fork = []
            db = DBSCAN(eps=1.0, min_samples=3).fit(ID[potential_fork,:])
            labels = db.labels_
            for l in set(labels):
                if l>=0:
                    current_id = np.where(labels==l)[0]
                    picked_id = current_id[np.where(s_fork[current_id]==np.max(s_fork[current_id]))[0][0]]
                    picked_fork.append(potential_fork[picked_id])
            db = DBSCAN(eps=1.0, min_samples=3).fit(ID[potential_tip,:])
            labels = db.labels_
            for l in set(labels):
                if l>=0:
                    current_id = np.where(labels==l)[0]
                    picked_id = current_id[np.where(s_tip[current_id]==np.min(s_tip[current_id]))[0][0]]
                    picked_tip.append(potential_tip[picked_id])

            if picked_tip.__len__() + picked_fork.__len__()>=2:
                tip_points.append(picked_tip)
                fork_points.append(picked_fork)

    plt.subplot(1,5,4)
    ax3 = plt.gca()
    for id in np.arange(tip_points.__len__()):
        picked_tip = np.array(tip_points[id])
        picked_fork = np.array(fork_points[id])
        ax3.plot(ID[picked_tip,1], ID[picked_tip,0], 'o', markerfacecolor='r',
                 markeredgecolor='k', markersize=14)
        ax3.plot(ID[picked_fork,1], ID[picked_fork,0], 'o', markerfacecolor='g',
                 markeredgecolor='k', markersize=14)
    ax3.invert_yaxis()
    plt.imshow(tip_img+np.min(tip_img),'gray')
    return tip_points, fork_points

# connect key points together using a customized distance metric
def connectPoints(tip_points,fork_points,connectivity_list,C,G,ID):
    tree_set = []
    # do for each connected image subgraph
    for tip_group, fork_group in zip(tip_points, fork_points):
        key_points = np.concatenate((tip_group,fork_group),axis=0)
        # key_points = np.array(tip_group)
        N = key_points.shape[0]
        metaG = nx.Graph()
        metaG.add_nodes_from(np.arange(N)) # make a weigthed undirected graph of key points
        path = [[[] for j in range(N)] for i in range(N)]
        for i in np.arange(N):
            for j in np.arange(i+1,N):
                path[i][j] = nx.dijkstra_path(G,key_points[i],key_points[j]) # get the shortest path from the image graph
                path[j][i] = path[i][j]
                path_length = nx.dijkstra_path_length(G,key_points[i],key_points[j])
                euclidean_distance = np.linalg.norm(ID[key_points[i],:]-ID[key_points[j],:])
                # divide the tangent distance by euclidean distance to favorite long strokes
                # metaG.add_weighted_edges_from([(i,j,path_length/euclidean_distance)])
                metaG.add_weighted_edges_from([(i,j,path_length)])

        T=nx.minimum_spanning_tree(metaG) # get the minimum spanning tree
        tree_set.append((T,path))
    return tree_set

        # for i in np.arange(N):
        #     key_connection[i,path_length[i,:]==np.min(path_length[i,:])] = 1
        # key_connection += key_connection.T # make it symmetric
        # key_connection = key_connection>0 + 0 # make it binary

        # temp_connection = key_connection.copy()
        # while True:
        #     temp_connection = np.dot(temp_connection,key_connection)>0 + 0 # calculate connection of higher path length
        #     for i in np.arange(N):
        #         center_points = np.where(temp_connection==1)[0]
        #         for cp in center_points:


def fitBspline(ID, connected_group, G, etf, row_, col_):
    point_cluster = []
    bspline_set = []
    for group in connected_group:
        if group.__len__() >= 3: # ignore small groups
            subG = nx.subgraph(G,group)
            while group.__len__() > 3: # Check if group is non-empty after strokes taken away
                point_cluster.append([])

                ## find the tip points
                xy = ID[group,:]
                temp_img = np.ones((row_,col_)) # create temp white image
                temp_img[xy[:,0],xy[:,1]] = 0 # set subgraph as a black pattern
                fig1 = plt.figure()
                ax1 = fig1.add_subplot(111)
                ax1.imshow(temp_img)
                plt.show()

                s = np.zeros((xy.shape[0],)) # initialize score
                dd = int(np.round((1-sum(sum(temp_img))/float(temp_img.size))*min(row_,col_))/2)
                for trial, d in enumerate([dd-1,dd,dd+1]):
                    for id, coord in enumerate(xy):
                        for r_ in np.arange(-d,d+1):
                            for c_ in np.arange(-d,d+1):
                                if r_!=0 or c_!=0:
                                    if coord[0]+r_>=0 and coord[0]+r_<row_ and coord[1]+c_>=0 and coord[1]+c_<col_:
                                        s[id] -= temp_img[coord[0]+r_,coord[1]+c_] # more white areas means closer to tip
                                    else:
                                        s[id] -= 1
                        if trial==0 and s[id]<-(2*d+1)**2+1: # if isolated point, remove
                            subG.remove_node(group[id])
                            group.remove(group[id])

                # temp_img[xy[:,0],xy[:,1]] = s
                # tip_img += temp_img

                s -= np.min(s) # set s to be non-negative
                s /= (np.max(s)+0.0000001) # set s to [0,1]

                ## cluster s to get the tips and fork points
                db = KMeans(n_clusters=3).fit(s[np.newaxis].T)
                labels = db.labels_
                u_labels = set(labels)
                sum_s = np.zeros((3,))
                for id, l in enumerate(u_labels):
                    sum_s[id] = np.average(s[labels==l])
                # potential_fork = np.array(group)[labels==list(u_labels)[np.where(sum_s==np.max(sum_s))[0][0]]]
                potential_tip = np.array(group)[labels==list(u_labels)[np.where(sum_s==np.min(sum_s))[0][0]]]
                # s_fork = s[labels==list(u_labels)[np.where(sum_s==np.max(sum_s))[0][0]]]
                s_tip = s[labels==list(u_labels)[np.where(sum_s==np.min(sum_s))[0][0]]]

                db = DBSCAN(eps=1.0, min_samples=3).fit(ID[potential_tip,:])
                labels = db.labels_
                picked_tip = []
                for l in set(labels):
                    if l>=0:
                        current_id = np.where(labels==l)[0]
                        picked_id = current_id[np.where(s_tip[current_id]==np.min(s_tip[current_id]))[0][0]]
                        picked_tip.append(potential_tip[picked_id])

                ## calculate distance between each pair of tip points and find the shortest distance
                key_points = np.array(picked_tip)
                N = key_points.shape[0]
                d = 1000000
                path = []
                for i in np.arange(N):
                    for j in np.arange(i+1,N):
                        try:
                            path_length = nx.dijkstra_path_length(subG,key_points[i],key_points[j])
                            if path_length<d:
                                d = path_length
                                path = nx.dijkstra_path(subG,key_points[i],key_points[j]) # get the shortest path from the image graph
                        except:
                            print "Oops!"

                ## remove all nodes close to the path
                for n in path:
                    if n in subG.nodes():
                        subG.remove_node(n)
                        group.remove(n)
                        point_cluster[-1].append(n)
                        r, c = ID[n,:].astype(int)
                        cos_theta = etf[r,c,1]
                        sin_theta = -etf[r,c,0]
                        # scan in one direction
                        k = 1
                        more = True
                        while more:
                            more = False
                            r_offset = int(np.round(sin_theta*k))
                            c_offset = int(np.round(cos_theta*k))

                            n1 = np.logical_and(ID[:,0]==r+r_offset,ID[:,1]==c+c_offset).nonzero()[0]
                            if n1.size>0: # if point is black, i.e., in the ID set
                                n1 = n1[0]
                                if n1 in group: # if in the current group
                                    t_n1 = etf[r+r_offset,c+c_offset,:]
                                    t = etf[r,c,:]
                                    if np.abs(np.dot(t_n1,t))>0.8: # if two tangent are aligned
                                        subG.remove_node(n1)
                                        group.remove(n1)
                                        point_cluster[-1].append(n1)
                                        more = True
                                        k += 1

                        # scan in the other direction
                        k = 1
                        more = True
                        while more:
                            more = False
                            r_offset = int(np.round(sin_theta*k))
                            c_offset = int(np.round(cos_theta*k))

                            n1 = np.logical_and(ID[:,0]==r-r_offset,ID[:,1]==c-c_offset).nonzero()[0]
                            if n1.size>0: # if point is black, i.e., in the ID set
                                n1 = n1[0]
                                if n1 in group: # if in the current group
                                    t_n1 = etf[r-r_offset,c-c_offset,:]
                                    t = etf[r,c,:]
                                    if np.abs(np.dot(t_n1,t))>0.8: # if two tangent are aligned
                                        subG.remove_node(n1)
                                        group.remove(n1)
                                        point_cluster[-1].append(n1)
                                        more = True
                                        k += 1
                if point_cluster[-1].__len__()>3:
                    try:
                        c,u0,X = fit_bspline(ID[point_cluster[-1],1], ID[point_cluster[-1],0], num_control_points=int(min(max(3,point_cluster[-1].__len__()/10.0),10)))
                        bspline_set.append([c,u0,X])
                    except:
                        what = 1

    return bspline_set





# convert point clusters to bsplines
def getParameter(ID,tree_set):
    par_set = []
    counter = 0
    for T,path in tree_set:
        edge_set = T.edges(data=True)
        for u,v,w in edge_set:
            xy = ID[path[u][v],:]
            if xy.shape[0]>=3:
                counter+=1
                c,u0,X = fit_bspline(xy[:,1], xy[:,0], num_control_points=min(max(3,xy.shape[0]/10),10))
                par_set.append([c,u0,X])

    # unique_labels = set(point_cluster_labels)
    # for l in unique_labels:
    #     if l>=0: # DBSCAN label=-1 are undetermined points
    #         xy = ID[point_cluster_labels==l,:]
    #         c,u,X = fit_bspline(xy[:,1], xy[:,0], num_control_points=min(max(3,xy.shape[0]/10),10))
    #         par_set.append([l,c,u,X])

    # Plot the results
    plt.subplot(1,5,5)
    ax1 = plt.gca()
    for par in par_set:
        bspline = par[0]
        u0 = par[1]
        x_plot = par[2]
        ax1.plot( *zip(*bspline.M(u0, x_plot).tolist()),linewidth =2)#, c=u0, cmap="jet", alpha=0.5 )
        ax1.plot(*zip(*x_plot), marker="o", alpha=0.3)
    # Since images render upside down
    ax1.invert_yaxis()
    # ax1.set_autoscale_on(False)
    return par_set

#######################################################################################################
#################################### main code ########################################################
#######################################################################################################
address = 'raw_image/'
filename = 'grayscale_image1.jpg'
img = cv2.imread(address + filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, None, fx=0.5, fy=0.5, )
row_, col_ = gray.shape
plt.subplot(1,5,1)
plt.imshow(gray,'gray')

debug = True
############################## The following section preprocess the image ##############################
fdog_img_address = 'fdog_test_data.out'
etf_address = 'etf_test_data.out'

if os.path.isfile(fdog_img_address+'.npy') and os.path.isfile(etf_address+'.npy') and not debug:
    fdog_img = np.load(fdog_img_address+'.npy')
    tan_ETF = np.load(etf_address+'.npy')
else:
    ## shrink the image to a pre-defined standard
    gray_ = gray.copy()

    ## Calculate tangent flow
    tan_ETF = edge_tangent_flow(gray_)

    ## FDoG loop
    fdog_loop = 3
    for count in np.arange(fdog_loop):
        ## Get FDoG Edge
        sigma_e = 1.0
        sigma_r = 1.6 #1.6
        sigma_m = 3.0 #3.0
        tau = 0.99 #0.99
        phi = 2.0 #2.0
        threshold = 1.5 #0.2
        fdog_img, f0, f1 = FDoGedge.getFDoGedge(tan_ETF.astype(np.float64),gray_.astype(np.float64),
                                        sigma_e,sigma_r,sigma_m,tau,phi,threshold)
        gray_[fdog_img<255]=0.0
        # tan_ETF = edge_tangent_flow(gray_)
    np.save(fdog_img_address, fdog_img)
    np.save(etf_address, tan_ETF)

lic_img = visualizeByLIC(tan_ETF)
plt.subplot(1,5,2)
plt.imshow(lic_img,'gray')
plt.subplot(1,5,3)
plt.imshow(fdog_img,'gray')

# ############################## The following section parameterizes the image ##############################
# ID = np.array(np.where(fdog_img==0)).T #all pixel indices
#
# file_address = 'processed_data/' + filename + '.pickle'
# if os.path.isfile(file_address):
#     with open(file_address) as f:
#         tree_set, tip_points, fork_points, G, connectivity_list, C, ID = pickle.load(f)
#     f.close()
#
#     plt.subplot(1,5,4)
#     ax3 = plt.gca()
#     for id in np.arange(tip_points.__len__()):
#         picked_tip = np.array(tip_points[id])
#         picked_fork = np.array(fork_points[id])
#         ax3.plot(ID[picked_tip,1], ID[picked_tip,0], 'o', markerfacecolor='r',
#                  markeredgecolor='k', markersize=14)
#         ax3.plot(ID[picked_fork,1], ID[picked_fork,0], 'o', markerfacecolor='g',
#                  markeredgecolor='k', markersize=14)
#     ax3.invert_yaxis()
#     plt.imshow(fdog_img,'gray')
#
# else:
#     ## Step 2: Create image graph
#     G = createGraph(ID,ID.shape[0],tan_ETF)
#     connectivity_list, C = calConnectivity(G,ID.shape[0])
#
#     # ## Step 3: Find key points ===THIS PART NEEDS IMPROVEMENT===
#     # tip_points, fork_points = findTipPoints(ID,connectivity_list,row_,col_)
#     #
#     # ## Step 4: Create minimum spanning tree of a metagraph from key points
#     # tree_set = connectPoints(tip_points,fork_points,connectivity_list,C,G,ID)
#
#     par_set = fitBspline(ID, connectivity_list, G, tan_ETF, row_, col_)
#     # Plot the results
#     plt.subplot(1,5,5)
#     ax1 = plt.gca()
#     for par in par_set:
#         bspline = par[0]
#         u0 = par[1]
#         x_plot = par[2]
#         ax1.plot( *zip(*bspline.M(u0, x_plot).tolist()),linewidth =2)#, c=u0, cmap="jet", alpha=0.5 )
#         ax1.plot(*zip(*x_plot), marker="o", alpha=0.3)
#     # Since images render upside down
#     ax1.invert_yaxis()
#     # ax1.set_autoscale_on(False)
#
#     ## Store everything
#     with open(file_address, 'w') as f:
#         pickle.dump([par_set, G, connectivity_list, C, ID], f)
#     f.close()
#
# ## Step 5: Convert the minimum spanning tree to bsplines
# par_set = getParameter(ID,tree_set)













# ID, point_cluster_labels = getCluster(fdog_img,tan_ETF)
# par_set = getParamter(ID,point_cluster_labels)





# par_img = drawFromParameter(par_set,fdog_img.shape[0],fdog_img.shape[1])
# plt.subplot(1,2,1)
# plt.imshow(fdog_img, 'gray')
# plt.subplot(1,2,2)
# plt.imshow(par_img, 'gray')
plt.show()
