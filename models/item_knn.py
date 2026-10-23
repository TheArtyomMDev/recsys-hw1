import numpy as np
import scipy.sparse as sp

from .base import BaseRecommender


def topk_rows(S, k):
    S = S.tocsr()
    cols, vals, indptr = [], [], [0]
    for i in range(S.shape[0]):
        lo, hi = S.indptr[i], S.indptr[i + 1]
        sel = np.argpartition(-S.data[lo:hi], k - 1)[:k] if hi - lo > k else slice(None)
        cols.append(S.indices[lo:hi][sel])
        vals.append(S.data[lo:hi][sel])
        indptr.append(indptr[-1] + len(cols[-1]))
    return sp.csr_matrix((np.concatenate(vals), np.concatenate(cols), np.asarray(indptr)), shape=S.shape)


class ItemKNN(BaseRecommender):
    name = "ItemKNN"

    def __init__(self, k_neighbors=100, shrink=10.0, alpha=0.5, weighting="binary", normalize=False):
        self.k_neighbors = int(k_neighbors)
        self.shrink = float(shrink)
        self.alpha = float(alpha)
        self.weighting = weighting
        self.normalize = bool(normalize)

    def fit(self, data):
        X = (data.csr if self.weighting == "binary" else data.csr_rating).astype(np.float32).tocsr()
        self.n_items, self._X = data.n_items, X
        G = (X.T @ X).tocsr()
        G.setdiag(0.0)
        G.eliminate_zeros()
        G = G.tocoo()
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel()) + 1e-8
        sim = G.data / (norms[G.row] ** (2 * self.alpha) * norms[G.col] ** (2 * (1 - self.alpha)) + self.shrink)
        W = topk_rows(sp.csr_matrix((sim.astype(np.float32), (G.row, G.col)), shape=G.shape), self.k_neighbors)
        if self.normalize:
            W = sp.diags(1.0 / (np.asarray(abs(W).sum(axis=1)).ravel() + 1e-8)) @ W
        self._WT = W.T.tocsr().astype(np.float32)
        return self

    def score(self, users):
        return (self._X[users] @ self._WT).toarray()
