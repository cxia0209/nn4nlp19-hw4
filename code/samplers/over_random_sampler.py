import torch
from torch.utils.data import Sampler


class OverRandomSampler(Sampler):
    r"""
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
        oversampling (sequence): oversample time of each item
    """

    def __init__(self, data_source, replacement=False, num_samples=None, oversampling=None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        self.oversampling = oversampling
        self.pseudo_id2id = []

        if self.num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if self.num_samples is None:
            self.num_samples = len(self.data_source)

        if self.oversampling is None:
            self.oversampling = [1] * self.num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        for i, t in enumerate(self.oversampling):
            self.pseudo_id2id += [i] * t

    def __iter__(self):
        n = len(self.pseudo_id2id)
        if self.replacement:
            pseudo_id_list = torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist()
        else:
            pseudo_id_list = torch.randperm(n).tolist()
        id_list = [self.pseudo_id2id[pid] for pid in pseudo_id_list]
        return iter(id_list)

    def __len__(self):
        return len(self.pseudo_id2id)
