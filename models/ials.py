from concurrent.futures import ThreadPoolExecutor

import numpy as np
from threadpoolctl import threadpool_limits

from .base import BaseRecommender


def solve_side(M, Y, reg, alpha, out, max_elems=2e7, n_jobs=8):
    f = Y.shape[1]
    YtY = Y.T @ Y + reg * np.eye(f)
    nnz = np.diff(M.indptr)
    out[nnz == 0] = 0.0

    def solve_chunk(rows, n):
        offs = M.indptr[rows][:, None] + np.arange(n)
        Yb = Y[M.indices[offs]]
        c1 = alpha * M.data[offs]
        A = YtY + np.matmul(Yb.transpose(0, 2, 1) * c1[:, None, :], Yb)
        out[rows] = np.linalg.solve(A, np.einsum("bnf,bn->bf", Yb, 1.0 + c1)[..., None])[..., 0]

    jobs = []
    for n in np.unique(nnz[nnz > 0]):
        rows = np.nonzero(nnz == n)[0]
        chunk = max(1, min(1024, int(max_elems // (n * f))))
        jobs += [(rows[s:s + chunk], n) for s in range(0, len(rows), chunk)]
    with threadpool_limits(1, "blas"), ThreadPoolExecutor(n_jobs) as ex:
        list(ex.map(lambda job: solve_chunk(*job), jobs))


class IALS(BaseRecommender):
    name = "iALS"

    def __init__(self, factors=64, reg=0.1, alpha=10.0, iterations=15, init_std=0.01, weighting="binary", seed=42):
        self.factors = int(factors)
        self.reg = float(reg)
        self.alpha = float(alpha)
        self.iterations = int(iterations)
        self.init_std = float(init_std)
        self.weighting = weighting
        self.seed = int(seed)

    def fit(self, data):
        X = (data.csr if self.weighting == "binary" else data.csr_rating).astype(np.float64).tocsr()
        Xt = X.T.tocsr()
        self.n_items = data.n_items
        rng = np.random.default_rng(self.seed)
        U = rng.normal(0.0, self.init_std, (data.n_users, self.factors))
        V = rng.normal(0.0, self.init_std, (data.n_items, self.factors))
        for _ in range(self.iterations):
            solve_side(X, V, self.reg, self.alpha, U)
            solve_side(Xt, U, self.reg, self.alpha, V)
        self._U, self._V = U.astype(np.float32), V.astype(np.float32)
        return self

    def score(self, users):
        return self._U[users] @ self._V.T
