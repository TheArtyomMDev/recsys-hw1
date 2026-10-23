import time

import numpy as np
import pandas as pd


def per_user_metrics(recs, relevant, k):
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    rows = []
    for rec, rel in zip(recs, relevant):
        hit = np.isin(rec[:k], rel)
        idcg = discounts[:min(len(rel), k)].sum()
        rows.append({"precision": hit.sum() / k, "recall": hit.sum() / len(rel), "ndcg": (hit * discounts).sum() / idcg})
    return pd.DataFrame(rows)


class Evaluator:
    def __init__(self, data, relevant, k=10):
        self.data, self.k = data, k
        self.users = np.array(sorted(relevant), dtype=np.int64)
        self.relevant = [relevant[u] for u in self.users]
        self.candidates = data.warm_items
        self.exclude = data.csr

    def evaluate(self, model):
        recs = model.recommend(self.users, k=self.k, exclude=self.exclude, candidates=self.candidates)
        pu = per_user_metrics(recs, self.relevant, self.k)
        res = {f"{m}@{self.k}": float(pu[m].mean()) for m in ["ndcg", "recall", "precision"]}
        res[f"coverage@{self.k}"] = len(np.unique(recs)) / len(self.candidates)
        return res

    def fit_and_evaluate(self, model):
        t0 = time.time()
        model.fit(self.data)
        return {**self.evaluate(model), "fit_time": time.time() - t0}
