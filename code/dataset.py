import torch
import os
import numpy as np
import torch
import torch.utils.data as D
from torch.utils.data import Sampler

from vocab import Vocabulary, BOS, EOS, PAD, UNK


def word_ids2string(ids, char_vocab):
    ret = []
    for i in ids:
        if isinstance(i, torch.Tensor):
            idx = i.item()
        else:
            idx = i
        if idx != char_vocab.pad and idx != char_vocab.bos and idx != char_vocab.eos:
            ret.append(char_vocab.i2w(idx))
    return ret


def feat_ids2string(ids, feat_vocab):
    ret = []
    for i in ids:
        idx = i.item()
        if idx != feat_vocab.unk:
            ret.append(idx)
    return ' '.join(feat_vocab.decode(ret))


def pos_ids2string(ids, pos_vocab):
    ret = []
    for i in ids:
        idx = i.item()
        if idx != pos_vocab.unk:
            ret.append(idx)
    return ' '.join(pos_vocab.decode(ret))


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


class MorphDataset(D.Dataset):

    def __init__(self, filenames, char_vocab=None, feat_vocab=None, 
                 pos_vocab=None, pos_sp=True, train=True, covered=False):
        super().__init__()
        if isinstance(filenames, list):
            self.filenames = filenames
        elif isinstance(filenames, str):
            self.filenames = [filenames]
        else:
            raise ValueError
        self.train = train
        if char_vocab is None or feat_vocab is None or pos_vocab is None:
            assert char_vocab is None and feat_vocab is None and pos_vocab is None  # should be None at the same time
        if char_vocab is None:  # if None, create new vocabs
            self.char_vocab = Vocabulary(unk=True, pad=True, bos=True, eos=True)
            self.feat_vocab = Vocabulary(unk=True)
            self.pos_vocab = Vocabulary(unk=True)
            self.m_char_vocab = self.char_vocab
        else:  # else, load existing vocabs
            self.char_vocab = char_vocab
            self.feat_vocab = feat_vocab
            self.pos_vocab = pos_vocab
            self.m_char_vocab = Vocabulary.from_vocab(char_vocab)

        self.raw_data = []
        self.data = []
        self.organized_data = []
        self.data_sizes = []
        self.pos_sp = pos_sp
        self.covered = covered
        self.build_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.organized_data[idx]

    def get_data_sizes(self):
        return self.data_sizes

    def get_vocabs(self):
        return self.char_vocab, self.feat_vocab, self.pos_vocab

    def get_m_voacb(self):
        return self.m_char_vocab

    def build_dataset(self):
        for sample in self.read_file():
            self.raw_data.append(sample)
            if self.covered:
                lang, lemma, feat, pos = sample
            else:
                lang, lemma, word, feat, pos = sample

            lemma_ids = self.char_vocab.encode(lemma, growth=self.train)
            if not self.covered:
                word_ids = self.char_vocab.encode(word, growth=self.train)
            feat_ids = self.feat_vocab.encode(feat, growth=self.train)
            pos_ids = self.pos_vocab.encode(pos, growth=self.train)

            m_lemma_ids = self.m_char_vocab.encode(lemma, growth=True)
            if not self.covered:
                m_word_ids = self.m_char_vocab.encode(word, growth=True)

            if self.covered:
                self.data.append((lang, lemma_ids, feat_ids, pos_ids, m_lemma_ids))
            else:
                self.data.append((lang, lemma_ids, word_ids, feat_ids, pos_ids, m_lemma_ids, m_word_ids))

        for sample in self.data:
            if self.covered:
                lang_id, lemma_ids, feat_ids, pos_ids, m_lemma_ids = sample
            else:
                lang_id, lemma_ids, word_ids, feat_ids, pos_ids, m_lemma_ids, m_word_ids = sample
            # process feats and poss
            ifeat_vec = []
            for i in self.feat_vocab.dic_i2w:
                if i != self.feat_vocab.unk:  # do not count unk feat
                    if i in feat_ids:
                        ifeat_vec.append(i)
                    else:
                        ifeat_vec.append(self.feat_vocab.unk)

            if self.pos_sp:
                ipos_vec = [p for p in pos_ids]
            else:
                ipos_vec = []
                for i in self.pos_vocab.dic_i2w:
                    if i != self.pos_vocab.unk:  # do not count unk pos
                        if i in pos_ids:
                            ipos_vec.append(i)
                        else:
                            ipos_vec.append(self.pos_vocab.unk)

            if self.covered:
                self.organized_data.append((np.array(lang_id, dtype=np.int64),
                                            np.array(lemma_ids, dtype=np.int64),
                                            np.array(len(lemma_ids), dtype=np.int64),
                                            np.array(ifeat_vec, dtype=np.int64),
                                            np.array(ipos_vec, dtype=np.int64),
                                            np.array(m_lemma_ids, dtype=np.int64)))
            else:
                self.organized_data.append((np.array(lang_id, dtype=np.int64),
                                            np.array(lemma_ids, dtype=np.int64),
                                            np.array(len(lemma_ids), dtype=np.int64),
                                            np.array(word_ids, dtype=np.int64),
                                            np.array(len(word_ids), dtype=np.int64),
                                            np.array(ifeat_vec, dtype=np.int64),
                                            np.array(ipos_vec, dtype=np.int64),
                                            np.array(m_lemma_ids, dtype=np.int64),
                                            np.array(m_word_ids, dtype=np.int64)))

    def read_file(self):
        for lang, filename in enumerate(self.filenames):
            lang = lang if self.train else 1
            with open(filename, 'r', encoding='utf-8') as fp:
                num = 0
                for line in fp.readlines():
                    if len(line.strip()) == 0:
                        continue
                    num += 1
                    if self.covered:
                        lemma, tags = line.strip().split('\t')
                    else:
                        lemma, word, tags = line.strip().split('\t')
                    tags = tags.split(';')
                    assert len(tags) > 1
                    if self.covered:
                        yield lang, list(lemma), tags[1:], tags[:1]
                    else:
                        yield lang, list(lemma), list(word), tags[1:], tags[:1]
                self.data_sizes.append(num)


class MorphDataloader(D.DataLoader):
    def __init__(self, dataset, left_padding=False, **kwargs):
        super(MorphDataloader, self).__init__(
            dataset=dataset,
            collate_fn=self.collate_fn,
            **kwargs
        )
        self.pad = self.dataset.char_vocab.pad
        self.covered = self.dataset.covered
        self.char_vocab_size = len(self.dataset.char_vocab)
        self.feat_vocab_size = len(self.dataset.feat_vocab)
        self.pos_vocab_size = len(self.dataset.pos_vocab)
        self.m_char_vocab_size = len(self.dataset.char_vocab)
        self.left_padding = left_padding  # left padding or not

    """
    Prepare for each batches
    return : 
        Tensor: padded lemma matrix
        Tensor: lemma lengths
        Tensor: padded words matrix
        Tensor: word lengths
        Tensor: one hot vectors for attributes
        Tensor: one hot vectors for part of speech tagging
    """
    def collate_fn(self, batches):
        if self.covered:
            langs, lemmas, lemma_lens, feats, poss, m_lemmas = tuple(zip(*batches))
        else:
            langs, lemmas, lemma_lens, words, word_lens, feats, poss, m_lemmas, m_words = tuple(zip(*batches))

        #padding
        padded_lemmas = self.zero_pad_concat(lemmas, self.pad, self.left_padding)
        if not self.covered:
            padded_words = self.zero_pad_concat(words, self.pad, self.left_padding)
        padded_m_lemmas = self.zero_pad_concat(m_lemmas, self.pad, self.left_padding)
        if not self.covered:
            padded_m_words = self.zero_pad_concat(m_words, self.pad, self.left_padding)
        
        nhot_feats = np.array(feats, dtype=np.int64)
        nhot_poss = np.array(poss, dtype=np.int64)

        np_langs = np.array(langs, dtype=np.int64)

        np_lemma_lens = np.array(lemma_lens, dtype=np.int64)
        if not self.covered:
            np_word_lens = np.array(word_lens, dtype=np.int64)

        if self.covered:
            return torch.from_numpy(np_langs), torch.from_numpy(padded_lemmas), torch.from_numpy(np_lemma_lens), torch.from_numpy(nhot_feats), torch.from_numpy(nhot_poss), torch.from_numpy(padded_m_lemmas),
        else:
            return torch.from_numpy(np_langs), torch.from_numpy(padded_lemmas), torch.from_numpy(np_lemma_lens), torch.from_numpy(padded_words), torch.from_numpy(np_word_lens), torch.from_numpy(nhot_feats), torch.from_numpy(nhot_poss), torch.from_numpy(padded_m_lemmas), torch.from_numpy(padded_m_words)

    # general padding function without sort
    def zero_pad_concat(self, inputs, pad_value, left_padding=False):
        max_t = max(inp.shape[0] for inp in inputs)
        shape = (len(inputs), max_t)
        input_mat = np.full(shape, pad_value, dtype=np.int64)
        if left_padding:
            for e, inp in enumerate(inputs):
                input_mat[e, -inp.shape[0]:] = inp
        else:
            for e, inp in enumerate(inputs):
                input_mat[e, :inp.shape[0]] = inp
        return input_mat


def test():
    train_dataset = MorphDataset('../data/german-train-low', pos_sp=True)
    dev_dataset = MorphDataset('../data/german-dev', pos_sp=True, train=False,
                               char_vocab=train_dataset.char_vocab, 
                               feat_vocab=train_dataset.feat_vocab,
                               pos_vocab=train_dataset.pos_vocab)
    dataloader = MorphDataloader(dev_dataset, left_padding=False, batch_size=1, shuffle=False)

    print("char_vocab len: %d" %len(dev_dataset.char_vocab))
    print("feat_vocab len: %d" %len(dev_dataset.feat_vocab))
    print("pos_vocab len: %d" %len(dev_dataset.pos_vocab))
    print("m_char_vocab len: %d" %len(dev_dataset.m_char_vocab))

    for i, batch in enumerate(dataloader):
        lemmas, lemma_lens, words, word_lens, feats, poss, m_lemmas = batch
        print(lemmas)
        print(lemma_lens)
        print(words)
        print(word_lens)
        print(feats)
        print(poss)
        print(m_lemmas)
        print(feat_ids2string(feats[0], dev_dataset.feat_vocab))
        print(pos_ids2string(poss[0], dev_dataset.pos_vocab))
        print(word_ids2string(lemmas[0], dev_dataset.char_vocab))
        print(word_ids2string(words[0], dev_dataset.char_vocab))
        print(word_ids2string(m_lemmas[0], dev_dataset.m_char_vocab))
        print("---------")
        if i == 5:
            break

if __name__ == '__main__':
    test()
