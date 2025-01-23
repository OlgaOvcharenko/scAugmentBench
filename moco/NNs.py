from fbpca import pca
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse as sps

from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
import time
import faiss

KNN = 20
APPROX = True

def reduce_dimensionality(X, dim_red_k=100):
    k = min((dim_red_k, X.shape[0], X.shape[1]))
    U, s, Vt = pca(X, k=k) # Automatically centers.
    return U[:, range(k)] * s[range(k)]

# Exact nearest neighbors search.
def nn(ds1, ds2, knn=KNN, metric_p=2):
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(n_neighbors=knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((a, b_i))

    return match

def nn_approx(ds1, ds2, norm=True, knn=KNN, metric='manhattan', n_trees=10):
    if norm:
        ds1 = normalize(ds1)
        ds2 = normalize(ds2) 

    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
    ind = np.array(ind)

    return ind


def mnn(ds1, ds2, knn=10, approx=True):
    # Find nearest neighbors in first direction.
    if approx:
        match1 = nn_approx(ds1, ds2, knn=knn)
    else:
        match1 = nn(ds1, ds2, knn=knn)

    # Find nearest neighbors in second direction.
    if approx:
        match2 = nn_approx(ds2, ds1, knn=knn)
    else:
        match2 = nn(ds2, ds1, knn=knn)

    # Compute mutual nearest neighbors.
    mutual = match1 & set([ (b, a) for a, b in match2 ])

    return mutual

def mnn_approx(X, Y, norm=False, knn=10, metric='manhattan'):
    if norm:
        X = normalize(X)
        Y = normalize(Y)

    f = X.shape[1]
    t1 = AnnoyIndex(f, metric)
    t2 = AnnoyIndex(f, metric)
    for i in range(len(X)):
        t1.add_item(i, X[i])
    for i in range(len(Y)):
        t2.add_item(i, Y[i])
    t1.build(10)
    t2.build(10)

    mnn_mat = np.bool8(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t2.get_nns_by_vector(item, knn) for item in X])[:, 1:]
    for i in range(len(sorted_mat)):
        mnn_mat[i,sorted_mat[i]] = True
    _ = np.bool8(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t1.get_nns_by_vector(item, knn) for item in Y])[:, 1:]
    for i in range(len(sorted_mat)):
        _[sorted_mat[i],i] = True
    mnn_mat = np.logical_and(_, mnn_mat)
    pairs = np.array([[x, y] for x, y in zip(*np.where(mnn_mat>0))])
    return pairs

def computeMNNs(datasets, uni_cnames, knn, norm=False, metric='manhattan'):
    mnn_pairs = []
    n_ds = len(datasets)
    for i in range(n_ds):
        for j in range(i+1, n_ds):   # 我tm怎么会之前在这里写个range(i, n_ds) ??
            mnn_ij = mnn_approx(datasets[i], datasets[j], knn=knn, norm=norm, metric=metric)
            bi_names = uni_cnames[i][mnn_ij[:, 0]]
            bj_names = uni_cnames[j][mnn_ij[:, 1]]
            mnn_pairs.append(np.vstack([bi_names, bj_names]).T)

    mnn_pairs = np.vstack(mnn_pairs)
    return mnn_pairs

def random_walk1(nns, pairs, steps=10):
    pairs_plus = []
    for p in pairs:
        x, y = p[0], p[1]
        pairs_plus.append((x, y))

        for i in range(steps):
            nx = np.random.choice(nns[x])
            ny = np.random.choice(nns[y])
            pairs_plus.append((nx, ny))

    # keep only unique pairs
    pairs_plus = [[p[0], p[1]] for p in set(pairs_plus)]
    pairs_plus = np.asarray(pairs_plus)  # to array 

    return pairs_plus

DEFAULT_SEED = 1234

def run_kmeans(x, nmb_clusters, verbose=False, seed=DEFAULT_SEED):
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    clus.seed = seed
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)

    return [int(n[0]) for n in I]