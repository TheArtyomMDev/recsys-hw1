import numpy as np


class BaseRecommender:
    name = "base"

    def fit(self, data):
        raise NotImplementedError

    def score(self, users):
        raise NotImplementedError

    def recommend(self, users, k=10, exclude=None, candidates=None, batch_size=2048):
        users = np.asarray(users, dtype=np.int64)
        out = np.empty((len(users), k), dtype=np.int64)
        drop = None
        if candidates is not None:
            drop = np.ones(self.n_items, dtype=bool)
            drop[candidates] = False
        for s in range(0, len(users), batch_size):
            batch = users[s:s + batch_size]
            scores = np.array(self.score(batch), dtype=np.float32)
            if drop is not None:
                scores[:, drop] = -np.inf
            if exclude is not None:
                known = exclude[batch].tocoo()
                scores[known.row, known.col] = -np.inf
            top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1, kind="stable")
            out[s:s + len(batch)] = np.take_along_axis(top, order, axis=1)
        return out
