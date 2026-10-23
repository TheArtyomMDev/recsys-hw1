import gzip
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

RATINGS_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz"
META_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz"
_price_re = re.compile(r"\d+(?:\.\d+)?")


def download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_ratings(path):
    df = pd.read_csv(path).rename(columns={"parent_asin": "item_id"})
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df[["user_id", "item_id", "rating", "ts"]]


def parse_price(x):
    if isinstance(x, (int, float)):
        return float(x)
    m = _price_re.search(str(x).replace(",", "")) if x is not None else None
    return float(m.group()) if m else np.nan


def load_meta(path, item_ids=None):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if item_ids is not None and d.get("parent_asin") not in item_ids:
                continue
            rows.append({
                "item_id": d.get("parent_asin"), "title": d.get("title"), "main_category": d.get("main_category"),
                "categories": list(d.get("categories") or []), "store": d.get("store"), "price": parse_price(d.get("price")),
                "has_description": bool(d.get("description")), "n_images": len(d.get("images") or []),
            })
    return pd.DataFrame(rows).drop_duplicates("item_id").reset_index(drop=True)


def clean_interactions(df):
    stats = {"исходно": len(df)}
    df = df.dropna(subset=["user_id", "item_id", "rating", "ts"])
    stats["без пропусков"] = len(df)
    df = df[(df["rating"] >= 1) & (df["rating"] <= 5)]
    stats["оценка в [1, 5]"] = len(df)
    df = df.sort_values("ts", kind="stable").drop_duplicates(["user_id", "item_id"], keep="last")
    stats["без дубликатов пар"] = len(df)
    return df.reset_index(drop=True), stats


def encode_ids(df):
    users = pd.Index(sorted(df["user_id"].unique()))
    items = pd.Index(sorted(df["item_id"].unique()))
    df = df.assign(user=users.get_indexer(df["user_id"]).astype(np.int32), item=items.get_indexer(df["item_id"]).astype(np.int32))
    return df, users, items


def add_day_index(df, t0):
    return df.assign(t=((df["ts"] - pd.Timestamp(t0)) // pd.Timedelta(days=1)).astype(np.int32))


def temporal_split(df, val_start, test_start):
    val_start, test_start = pd.Timestamp(val_start), pd.Timestamp(test_start)
    parts = df[df["ts"] < val_start], df[(df["ts"] >= val_start) & (df["ts"] < test_start)], df[df["ts"] >= test_start]
    return tuple(p.reset_index(drop=True) for p in parts)


def interactions_to_csr(df, n_users, n_items, values=None):
    data = np.ones(len(df), dtype=np.float32) if values is None else df[values].to_numpy(dtype=np.float32)
    m = sp.csr_matrix((data, (df["user"].to_numpy(), df["item"].to_numpy())), shape=(n_users, n_items))
    m.sum_duplicates()
    if values is None:
        m.data[:] = 1.0
    return m


@dataclass
class TrainData:
    df: pd.DataFrame
    n_users: int
    n_items: int
    t_rec: int
    item_meta: pd.DataFrame | None = None
    _csr: sp.csr_matrix | None = field(default=None, repr=False)
    _csr_rating: sp.csr_matrix | None = field(default=None, repr=False)

    @property
    def csr(self):
        if self._csr is None:
            self._csr = interactions_to_csr(self.df, self.n_users, self.n_items)
        return self._csr

    @property
    def csr_rating(self):
        if self._csr_rating is None:
            self._csr_rating = interactions_to_csr(self.df, self.n_users, self.n_items, "rating")
        return self._csr_rating

    @property
    def warm_items(self):
        return np.unique(self.df["item"].to_numpy())


def build_eval_set(train_df, eval_df, min_rating=4.0):
    stats = {"строк в окне": len(eval_df), "пользователей в окне": int(eval_df["user"].nunique())}
    cold_user = ~eval_df["user"].isin(set(train_df["user"]))
    stats["строк холодных пользователей"] = int(cold_user.sum())
    stats["холодных пользователей"] = int(eval_df.loc[cold_user, "user"].nunique())
    e = eval_df[~cold_user]
    cold_item = ~e["item"].isin(set(train_df["item"]))
    stats["строк с холодными товарами"] = int(cold_item.sum())
    e = e[~cold_item]
    stats["тёплых строк"] = len(e)
    pos = e[e["rating"] >= min_rating]
    stats["из них положительных"] = len(pos)
    relevant = {int(u): g["item"].to_numpy() for u, g in pos.groupby("user")}
    stats["оцениваемых пользователей"] = len(relevant)
    stats["релевантных на пользователя"] = round(len(pos) / max(len(relevant), 1), 3)
    return relevant, stats
