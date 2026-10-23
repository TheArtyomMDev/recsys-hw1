from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class FeatureSpace:
    user_csr: sp.csr_matrix
    item_csr: sp.csr_matrix
    n_features: int
    groups: dict

    def describe(self):
        return pd.Series({g: e - s for g, (s, e) in self.groups.items()}, name="size").to_frame().T


class Builder:
    def __init__(self):
        self.offset, self.groups, self.rows, self.cols, self.vals = 0, {}, [], [], []

    def onehot(self, name, codes, n_values):
        mask = codes >= 0
        self.multihot(name, np.flatnonzero(mask), codes[mask], np.ones(int(mask.sum())), n_values)

    def multihot(self, name, rows, codes, vals, n_values):
        self.rows.append(np.asarray(rows, dtype=np.int64))
        self.cols.append(self.offset + np.asarray(codes, dtype=np.int64))
        self.vals.append(np.asarray(vals, dtype=np.float32))
        self.groups[name] = (self.offset, self.offset + n_values)
        self.offset += n_values


def bin_codes(values, edges, unknown=None):
    codes, n = np.digitize(values, edges).astype(np.int64), len(edges) + 1
    if unknown is not None:
        codes[unknown], n = n, n + 1
    return codes, n


def vocab_codes(values, min_count):
    s = [None if pd.isna(x) else x for x in values]
    counts = Counter(x for x in s if x is not None)
    vocab = {v: i for i, v in enumerate(sorted(v for v, c in counts.items() if c >= min_count))}
    other, unknown = len(vocab), len(vocab) + 1
    return np.array([unknown if x is None else vocab.get(x, other) for x in s]), len(vocab) + 2


def build_features(data, use_item_meta=True, use_user_meta=True, min_category_count=20, min_store_count=10, n_price_bins=8):
    df, n_users, n_items, meta = data.df, data.n_users, data.n_items, data.item_meta
    b = Builder()
    u_arr, i_arr, r_arr = df["user"].to_numpy(), df["item"].to_numpy(), df["rating"].to_numpy()
    user_cnt, item_cnt = np.bincount(u_arr, minlength=n_users), np.bincount(i_arr, minlength=n_items)
    rating_edges = [3.0, 3.5, 4.0, 4.5]

    b.onehot("user_id", np.arange(n_users), n_users)
    if use_user_meta:
        b.onehot("user_activity_bin", *bin_codes(user_cnt, [1, 2, 3, 5, 10, 20, 50]))
        mean_r = np.divide(np.bincount(u_arr, weights=r_arr, minlength=n_users), user_cnt, out=np.zeros(n_users), where=user_cnt > 0)
        b.onehot("user_mean_rating_bin", *bin_codes(mean_r, rating_edges, user_cnt == 0))
        recency = (data.t_rec - df.groupby("user")["t"].max().reindex(range(n_users)).to_numpy()).astype(float)
        b.onehot("user_recency_bin", *bin_codes(np.nan_to_num(recency), [30, 90, 180, 365, 730], np.isnan(recency)))

    b.onehot("item_id", np.arange(n_items), n_items)
    item_tokens, price, price_edges = None, None, None
    if use_item_meta and meta is not None:
        m = meta.reindex(range(n_items))
        b.onehot("item_main_category", *vocab_codes(m["main_category"], 5))
        cats = m["categories"].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
        tok_counts = Counter(t for toks in cats for t in set(toks))
        vocab = {t: i for i, t in enumerate(sorted(t for t, c in tok_counts.items() if c >= min_category_count))}
        item_tokens = [sorted({vocab[t] for t in toks if t in vocab}) for toks in cats]
        rows = [i for i, toks in enumerate(item_tokens) for _ in toks]
        cols = [t for toks in item_tokens for t in toks]
        vals = [1.0 / len(toks) for toks in item_tokens for _ in toks]
        b.multihot("item_category_token", rows, cols, vals, len(vocab))
        b.onehot("item_store", *vocab_codes(m["store"], min_store_count))
        price = m["price"].to_numpy(dtype=float)
        price_edges = np.unique(np.quantile(price[np.isfinite(price)], np.linspace(0, 1, n_price_bins + 1)[1:-1]))
        b.onehot("item_price_bin", *bin_codes(np.nan_to_num(price), price_edges, ~np.isfinite(price)))
    b.onehot("item_pop_bin", *bin_codes(item_cnt, [1, 3, 6, 11, 21, 51, 101, 301]))
    if use_item_meta:
        imean = np.divide(np.bincount(i_arr, weights=r_arr, minlength=n_items), item_cnt, out=np.zeros(n_items), where=item_cnt > 0)
        b.onehot("item_mean_rating_bin", *bin_codes(imean, rating_edges, item_cnt == 0))
        age = (data.t_rec - df.groupby("item")["t"].min().reindex(range(n_items)).to_numpy()).astype(float)
        b.onehot("item_age_bin", *bin_codes(np.nan_to_num(age), [30, 90, 180, 365, 730, 1460], np.isnan(age)))

    if use_user_meta and item_tokens is not None:
        rows, cols, vals = [], [], []
        for user, items in df.groupby("user")["item"]:
            cnt = Counter(t for item in items.to_numpy() for t in item_tokens[item])
            total = sum(cnt.values())
            for tok, c in cnt.items():
                rows.append(user), cols.append(tok), vals.append(c / total)
        b.multihot("user_category_profile", rows, cols, vals, len(vocab))
        up = df.assign(p=price[i_arr]).groupby("user")["p"].mean().reindex(range(n_users)).to_numpy(dtype=float)
        b.onehot("user_price_bin", *bin_codes(np.nan_to_num(up), price_edges, ~np.isfinite(up)))

    rows, cols, vals = np.concatenate(b.rows), np.concatenate(b.cols), np.concatenate(b.vals)
    is_user = np.zeros(b.offset, dtype=bool)
    for g, (s, e) in b.groups.items():
        is_user[s:e] = g.startswith("user_")
    um = is_user[cols]
    user_csr = sp.csr_matrix((vals[um], (rows[um], cols[um])), shape=(n_users, b.offset), dtype=np.float32)
    item_csr = sp.csr_matrix((vals[~um], (rows[~um], cols[~um])), shape=(n_items, b.offset), dtype=np.float32)
    user_csr.sum_duplicates()
    item_csr.sum_duplicates()
    return FeatureSpace(user_csr, item_csr, b.offset, b.groups)
