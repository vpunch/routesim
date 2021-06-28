import random
import numpy as np
import torch

from functools import cache

from .constants import INFTY

import networkx as nx


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stack_rows(rows):
    # [[[...], [...], [...]]
    #  [[...], [...], [...]]]

    # [[...  [...  [... ]
    #   ...], ...], ...]

    res = [[] for _ in range(len(rows[0]))]
    for row in rows:
        for i, emb in enumerate(row):
            res[i].append(emb)

    return list(map(np.array, res))


@cache
def get_multi_col(name, num):
    return [f'{name}_' + str(i) for i in range(num)]

# dqn_pretrain
def make_batches(size, batch_size):
    num_batches = int(np.ceil(size / float(batch_size)))
    for i in range(0, num_batches):
        yield (i * batch_size, min(size, (i + 1) * batch_size))

# dqn_pretrain
def stack_batch(batch):
    if type(batch[0]) == dict:
        return stack_batch_dict(batch)
    else:
        return stack_batch_list(batch)

# dqn_pretrain
def stack_batch_dict(batch):
    ss = {}
    for k in batch[0].keys():
        ss[k] = np.vstack([b[k] for b in batch])
        if ss[k].shape[1] == 1:
            ss[k] = ss[k].flatten()
    return ss

# dqn_pretrain
def stack_batch_list(batch):
    n = len(batch[0])
    ss = [None]*n
    for i in range(n):
        ss[i] = np.vstack([b[i] for b in batch])
        if ss[i].shape[1] == 1:
            ss[i] = ss[i].flatten()
    return ss



def delta(i: int, n: int):
    if i >= n:
        raise Exception('Action index is out of bounds')
    d = np.zeros(n)
    d[i] = 1
    return d


def uni(n):
    return np.full(n, 1.0/n)


def softmax(x, t=1.0):
    ax = np.array(x) / t
    ax -= np.amax(ax)
    e = np.exp(ax)
    sum = np.sum(e)
    if sum == 0:
        return uni(len(ax))
    return e / np.sum(e, axis=0)


def sample_distr(distr) -> int:
    return np.random.choice(np.arange(len(distr)), p=distr)


def soft_argmax(arr, t=1.0) -> int:
    return sample_distr(softmax(arr, t=t))
