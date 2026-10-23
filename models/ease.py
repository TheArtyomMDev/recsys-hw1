import numpy as np
import scipy.linalg

from .base import BaseRecommender


class EASE(BaseRecommender):
    name = "EASE"

    def __init__(self, reg=500.0, weighting="binary"):
        self.reg = float(reg)
        self.weighting = weighting

    def fit(self, data):
        X = (data.csr if self.weighting == "binary" else data.csr_rating).astype(np.float32).tocsr()
        self.n_items, self._X = data.n_items, X
        G = (X.T @ X).toarray()
        diag = np.diag_indices(G.shape[0])
        G[diag] += np.float32(self.reg)
        P = scipy.linalg.inv(G, overwrite_a=True, check_finite=False)
        B = P / (-np.diag(P))[None, :]
        B[diag] = 0.0
        self._B = B
        return self

    def score(self, users):
        return np.asarray(self._X[users] @ self._B)
