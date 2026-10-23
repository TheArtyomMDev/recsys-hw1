import numpy as np
import torch
import torch.nn.functional as F

from recsys_utils.features import build_features
from recsys_utils.torch_utils import CSRBag

from .base import BaseRecommender


class FactorizationMachine(BaseRecommender):
    name = "FM"

    def __init__(self, factors=32, lr=0.005, reg=1e-3, epochs=15, batch_size=4096, n_neg=1,
                 use_item_meta=True, use_user_meta=True, min_rating=1.0, init_std=0.01, seed=42):
        self.factors = int(factors)
        self.lr = float(lr)
        self.reg = float(reg)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.n_neg = int(n_neg)
        self.use_item_meta = bool(use_item_meta)
        self.use_user_meta = bool(use_user_meta)
        self.min_rating = float(min_rating)
        self.init_std = float(init_std)
        self.seed = int(seed)

    def _terms(self, bag, rows):
        idx, offsets, vals = bag.gather(rows)
        lin = F.embedding_bag(idx, self._w, offsets, per_sample_weights=vals, mode="sum").squeeze(1)
        s = F.embedding_bag(idx, self._V, offsets, per_sample_weights=vals, mode="sum")
        q = F.embedding_bag(idx, self._V ** 2, offsets, per_sample_weights=vals ** 2, mode="sum").sum(1)
        lsq = F.embedding_bag(idx, self._w ** 2, offsets, per_sample_weights=vals ** 2, mode="sum").squeeze(1)
        return lin, s, q, lsq

    def fit(self, data):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        fs = build_features(data, self.use_item_meta, self.use_user_meta)
        self.feature_space, self.n_items = fs, data.n_items
        self._w0 = torch.nn.Parameter(torch.zeros(1))
        self._w = torch.nn.Parameter(torch.zeros(fs.n_features, 1))
        self._V = torch.nn.Parameter(torch.randn(fs.n_features, self.factors) * self.init_std)
        ubag, ibag = CSRBag(fs.user_csr), CSRBag(fs.item_csr)
        df = data.df[data.df["rating"] >= self.min_rating]
        users, items = df["user"].to_numpy().astype(np.int64), df["item"].to_numpy().astype(np.int64)
        warm, csr, n_neg = data.warm_items, data.csr, self.n_neg
        opt = torch.optim.Adam([self._w0, self._w, self._V], lr=self.lr)
        self.loss_history = []
        for _ in range(self.epochs):
            perm, total, n_batches = rng.permutation(len(users)), 0.0, 0
            for s in range(0, len(users), self.batch_size):
                idx = perm[s:s + self.batch_size]
                u, i = users[idx], items[idx]
                j = rng.choice(warm, size=(len(idx), n_neg))
                u_rep = np.repeat(u, n_neg)
                for _ in range(3):
                    clash = np.asarray(csr[u_rep, j.ravel()]).ravel() > 0
                    if not clash.any():
                        break
                    j.ravel()[clash] = rng.choice(warm, size=int(clash.sum()))
                _, s_u, q_u, lsq_u = self._terms(ubag, torch.as_tensor(u))
                lin_i, s_i, q_i, lsq_i = self._terms(ibag, torch.as_tensor(i))
                lin_j, s_j, q_j, lsq_j = self._terms(ibag, torch.as_tensor(j.ravel()))
                x_ui = lin_i + (s_u * s_i).sum(1) + 0.5 * ((s_i ** 2).sum(1) - q_i)
                x_uj = lin_j + (s_u.repeat_interleave(n_neg, 0) * s_j).sum(1) + 0.5 * ((s_j ** 2).sum(1) - q_j)
                bpr = -F.logsigmoid(x_ui.repeat_interleave(n_neg) - x_uj).mean()
                reg = self.reg * (q_u.mean() + q_i.mean() + q_j.mean() + lsq_u.mean() + lsq_i.mean() + lsq_j.mean())
                opt.zero_grad()
                (bpr + reg).backward()
                opt.step()
                total, n_batches = total + bpr.item(), n_batches + 1
            self.loss_history.append(total / n_batches)
        with torch.no_grad():
            S_items, T_items, S_users = [], [], []
            for s in range(0, data.n_items, 8192):
                lin_i, s_i, q_i, _ = self._terms(ibag, torch.arange(s, min(s + 8192, data.n_items)))
                S_items.append(s_i)
                T_items.append(lin_i + 0.5 * ((s_i ** 2).sum(1) - q_i))
            for s in range(0, data.n_users, 8192):
                S_users.append(self._terms(ubag, torch.arange(s, min(s + 8192, data.n_users)))[1])
        self._S_items = torch.cat(S_items).numpy()
        self._T_items = torch.cat(T_items).numpy()
        self._S_users = torch.cat(S_users).numpy()
        return self

    def score(self, users):
        return self._T_items[None, :] + self._S_users[users] @ self._S_items.T
