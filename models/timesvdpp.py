import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from recsys_utils.torch_utils import CSRBag

from .base import BaseRecommender


class TimeSVDpp(BaseRecommender):
    name = "timeSVD++"

    def __init__(self, factors=32, lr=0.005, lr_decay=0.9, epochs=20, batch_size=2048, reg_bias=0.02, reg_factor=0.05,
                 reg_alpha=0.1, reg_day=1.0, beta=0.4, n_bins=30, use_time=True, day_bias=True, factor_drift=False,
                 init_std=0.01, seed=42):
        self.factors = int(factors)
        self.lr = float(lr)
        self.lr_decay = float(lr_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.reg_bias = float(reg_bias)
        self.reg_factor = float(reg_factor)
        self.reg_alpha = float(reg_alpha)
        self.reg_day = float(reg_day)
        self.beta = float(beta)
        self.n_bins = int(n_bins)
        self.use_time = bool(use_time)
        self.day_bias = bool(day_bias)
        self.factor_drift = bool(factor_drift)
        self.init_std = float(init_std)
        self.seed = int(seed)

    def _dev(self, t, users):
        d = t.astype(np.float64) - self._t_u[users]
        return (np.sign(d) * np.abs(d) ** self.beta / self._dev_scale).astype(np.float32)

    def _bin(self, t):
        return np.clip(np.floor((t - self._t_min) / self._bin_width).astype(np.int64), 0, self.n_bins - 1)

    def _implicit(self, users):
        idx, offsets, vals = self._nbag.gather(users)
        return F.embedding_bag(idx, self._y, offsets, per_sample_weights=vals, mode="sum")

    def _predict(self, u, i, dev, bins, days):
        b_user, b_item = self._bu[u], self._bi[i]
        if self.use_time:
            b_user = b_user + self._alpha_u[u] * dev
            if self.day_bias:
                b_user = b_user + self._b_ut[days]
            b_item = b_item + self._bi_bin[i * self.n_bins + bins]
        z = self._p[u] + self._implicit(u)
        if self.use_time and self.factor_drift:
            z = z + self._alpha_uk[u] * dev[:, None]
        return self._mu + b_user + b_item + (self._q[i] * z).sum(1), z

    def fit(self, data):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        df, n_users, n_items, k = data.df, data.n_users, data.n_items, self.factors
        self.n_items = n_items
        u_np, i_np = df["user"].to_numpy().astype(np.int64), df["item"].to_numpy().astype(np.int64)
        t_np, r_np = df["t"].to_numpy().astype(np.int64), df["rating"].to_numpy().astype(np.float32)
        self._mu = float(r_np.mean())
        self._t_min = int(t_np.min())
        self._bin_width = (int(t_np.max()) - self._t_min + 1) / self.n_bins
        self._t_u = pd.Series(t_np).groupby(u_np).mean().reindex(range(n_users)).fillna(data.t_rec).to_numpy(dtype=np.float64)
        self._dev_scale = 1.0
        self._dev_scale = float(np.abs(self._dev(t_np, u_np)).max()) or 1.0
        day_np = pd.factorize(pd.Series(u_np * 100000 + t_np))[0].astype(np.int64) + 1
        csr = data.csr.tocsr()
        nnz = np.diff(csr.indptr)
        weights = np.repeat(1.0 / np.sqrt(np.maximum(nnz, 1)), nnz).astype(np.float32)
        self._nbag = CSRBag(sp.csr_matrix((weights, csr.indices, csr.indptr), shape=csr.shape))

        std = self.init_std
        self._bu, self._bi = torch.nn.Parameter(torch.zeros(n_users)), torch.nn.Parameter(torch.zeros(n_items))
        self._bi_bin = torch.nn.Parameter(torch.zeros(n_items * self.n_bins))
        self._alpha_u = torch.nn.Parameter(torch.zeros(n_users))
        self._b_ut = torch.nn.Parameter(torch.zeros(int(day_np.max()) + 1))
        self._p = torch.nn.Parameter(torch.randn(n_users, k) * std)
        self._q = torch.nn.Parameter(torch.randn(n_items, k) * std)
        self._y = torch.nn.Parameter(torch.randn(n_items, k) * std)
        self._alpha_uk = torch.nn.Parameter(torch.zeros(n_users, k))
        params = [self._bu, self._bi, self._bi_bin, self._alpha_u, self._b_ut, self._p, self._q, self._y, self._alpha_uk]
        opt = torch.optim.SGD(params, lr=self.lr)

        dev_np, bins_np = self._dev(t_np, u_np), self._bin(t_np)
        self.loss_history = []
        for _ in range(self.epochs):
            perm, total = rng.permutation(len(u_np)), 0.0
            for s in range(0, len(u_np), self.batch_size):
                idx = perm[s:s + self.batch_size]
                u, i = torch.as_tensor(u_np[idx]), torch.as_tensor(i_np[idx])
                dev, bins, days = torch.as_tensor(dev_np[idx]), torch.as_tensor(bins_np[idx]), torch.as_tensor(day_np[idx])
                pred, z = self._predict(u, i, dev, bins, days)
                sq = ((pred - torch.as_tensor(r_np[idx])) ** 2).sum()
                reg = self.reg_bias * ((self._bu[u] ** 2).sum() + (self._bi[i] ** 2).sum())
                reg = reg + self.reg_factor * ((self._p[u] ** 2).sum() + (self._q[i] ** 2).sum() + (z ** 2).sum())
                if self.use_time:
                    reg = reg + self.reg_bias * (self._bi_bin[i * self.n_bins + bins] ** 2).sum()
                    reg = reg + self.reg_alpha * (self._alpha_u[u] ** 2).sum()
                    if self.day_bias:
                        reg = reg + self.reg_day * (self._b_ut[days] ** 2).sum()
                    if self.factor_drift:
                        reg = reg + self.reg_alpha * (self._alpha_uk[u] ** 2).sum()
                opt.zero_grad()
                (sq + reg).backward()
                opt.step()
                total += sq.item()
            self.loss_history.append(total / len(u_np))
            for g in opt.param_groups:
                g["lr"] *= self.lr_decay

        with torch.no_grad():
            b_item = self._bi.clone()
            if self.use_time:
                b_item = b_item + self._bi_bin.view(n_items, self.n_bins)[:, int(self._bin(np.array([data.t_rec]))[0])]
            dev_rec = torch.as_tensor(self._dev(np.full(n_users, data.t_rec), np.arange(n_users)))
            b_user = self._bu + self._alpha_u * dev_rec if self.use_time else self._bu.clone()
            Z = []
            for s in range(0, n_users, 8192):
                rows = torch.arange(s, min(s + 8192, n_users))
                z = self._p[rows] + self._implicit(rows)
                if self.use_time and self.factor_drift:
                    z = z + self._alpha_uk[rows] * dev_rec[rows, None]
                Z.append(z)
        self._item_bias, self._user_bias = b_item.numpy(), b_user.numpy()
        self._Q, self._Z = self._q.detach().numpy(), torch.cat(Z).numpy()
        return self

    def predict_ratings(self, users, items):
        return self._mu + self._user_bias[users] + self._item_bias[items] + np.einsum("ij,ij->i", self._Z[users], self._Q[items])

    def score(self, users):
        return self._mu + self._user_bias[users][:, None] + self._item_bias[None, :] + self._Z[users] @ self._Q.T
