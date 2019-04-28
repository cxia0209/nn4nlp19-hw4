import random
import torch
from torch.utils.data import Sampler


class OverBalancedRandomSampler(Sampler):
    r"""
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
        oversampling (sequence): oversample time of each item
    """

    def __init__(self, data_source, replacement=False, num_samples=None, oversampling=None, over_weight=1):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        self.oversampling = oversampling
        self.over_weight = over_weight
        self.true_ids = []
        self.false_ids = []
        self.pseudo_trueid2id = []
        self.pseudo_falseid2id = []

        if self.num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if self.oversampling is None:
            raise ValueError("oversampling should not be None")

        if not all([isinstance(x, bool) for x in self.oversampling]):
            raise ValueError("oversampling should be all bool")

        if len(data_source) != len(oversampling):
            raise ValueError("length of oversampling should be the same as the length of data_source")

        num_true = sum(self.oversampling)
        num_false = len(self.oversampling) - num_true
        if num_true <= 0 or num_false <= 0:
            raise ValueError("oversampling should contain both true and false, "
                             "but got num_true={}, num_false={}".format(num_true, num_false))

        if not isinstance(self.over_weight, int) or self.over_weight <= 0:
            raise ValueError("over_weight should be a positive integeral "
                             "value, but got over_weight={}".format(self.over_weight))

        if self.num_samples is None:
            self.num_samples = 2 * num_true * over_weight
        
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))
        if self.num_samples < over_weight * num_true:
            raise ValueError("num_samples should be greater or equal to over_weight * num_true")

        for i, b in enumerate(self.oversampling):
            if b:
                self.true_ids.append(i)
                self.pseudo_trueid2id += [i] * over_weight
            else:
                self.false_ids.append(i)
                self.pseudo_falseid2id.append(i)

        self.false_start = 0

    def __iter__(self):
        n_true = len(self.pseudo_trueid2id)
        n_false = len(self.pseudo_falseid2id)
        if self.replacement:
            pseudo_true_id_list = torch.randint(high=n_true, size=(n_true,), dtype=torch.int64).tolist()
            pseudo_false_id_list = torch.randint(high=n_false, size=(self.num_samples - n_true,), dtype=torch.int64).tolist()
        else:
            pseudo_true_id_list = torch.randperm(n_true).tolist()
            if self.false_start == 0:
                self.static_pseudo_false_id_list = torch.randperm(n_false).tolist()
            remaining = n_true
            pseudo_false_id_list = []
            while remaining > 0:
                new_end = min(self.false_start + remaining, n_false)
                pseudo_false_id_list += self.static_pseudo_false_id_list[self.false_start: new_end]
                remaining -= (new_end - self.false_start)
                if remaining > 0:
                    self.static_pseudo_false_id_list = torch.randperm(n_false).tolist()
                self.false_start = new_end % n_false

        true_id_list = [self.pseudo_trueid2id[pid] for pid in pseudo_true_id_list]
        false_id_list = [self.pseudo_falseid2id[pid] for pid in pseudo_false_id_list]
        id_list = true_id_list + false_id_list
        random.shuffle(id_list)
        return iter(id_list)

    def __len__(self):
        return self.num_samples
