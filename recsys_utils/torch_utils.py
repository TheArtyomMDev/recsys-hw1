import numpy as np
import torch


class CSRBag:
    def __init__(self, csr):
        csr = csr.tocsr()
        self.indptr = torch.as_tensor(csr.indptr.astype(np.int64))
        self.indices = torch.as_tensor(csr.indices.astype(np.int64))
        self.data = torch.as_tensor(csr.data.astype(np.float32))

    def gather(self, rows):
        starts = self.indptr[rows]
        lengths = self.indptr[rows + 1] - starts
        offsets = torch.cumsum(lengths, 0) - lengths
        pos = torch.arange(int(lengths.sum())) - torch.repeat_interleave(offsets, lengths) + torch.repeat_interleave(starts, lengths)
        return self.indices[pos], offsets, self.data[pos]
