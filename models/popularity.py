import numpy as np

from .base import BaseRecommender


class Popularity(BaseRecommender):
    name = "Popularity"

    def fit(self, data):
        self.n_items = data.n_items
        self._scores = np.bincount(data.df["item"].to_numpy(), minlength=data.n_items).astype(np.float32)
        return self

    def score(self, users):
        return np.tile(self._scores, (len(users), 1))
