"""Artificial sparse associative memory, independent of task labels/oracles.

The supplied inductive bias is explicit: enumerate subsets of <=3 input bits,
group demonstrations by their projected values, and retrieve empirical labels.
A neural network learns ONLY how much to trust each subset from its statistics.
This is not a learned memory implementation or an unrestricted program inducer.
"""
from functools import lru_cache
from itertools import combinations
import numpy as np

FEATURE_NAMES = ['minority_fraction', 'pair_disagreement', 'coverage', 'singleton_fraction',
                 'mean_bin_entropy', 'arity_fraction', 'log_support_per_bin', 'class_imbalance',
                 'log_support_size', 'observed_bin_fraction']


@lru_cache(maxsize=32)
def masks(n, max_arity=3):
    if not 1 <= n <= 16 or not 1 <= max_arity <= 3:
        raise ValueError('Supported widths are 1..16 and maximum arity 1..3.')
    subsets = [ix for k in range(1, min(n, max_arity) + 1) for ix in combinations(range(n), k)]
    codes = np.zeros((n, len(subsets)), dtype=np.int64)
    for h, ix in enumerate(subsets):
        codes[list(ix), h] = 1 << np.arange(len(ix))
    return subsets, codes


def encode(support_x, support_y, query_x, max_arity=3):
    sx, sy, qx = np.asarray(support_x), np.asarray(support_y), np.asarray(query_x)
    if qx.ndim != 2 or sx.ndim != 2 or sx.shape[1] != qx.shape[1] or sy.shape != (len(sx),) or not len(sx):
        raise ValueError('Nonempty rectangular support and matching query width are required.')
    if any(np.any((a != 0) & (a != 1)) for a in [sx, sy, qx]):
        raise ValueError('Only binary inputs and labels are supported.')
    sx, sy, qx = sx.astype(np.int64), sy.astype(np.int64), qx.astype(np.int64)
    subsets, codes = masks(sx.shape[1], max_arity)
    h = len(subsets); k = len(sx)
    bins = (sx @ codes).T
    flat = (bins + 8 * np.arange(h)[:, None]).ravel()
    counts = np.bincount(flat, minlength=h * 8).reshape(h, 8).astype(np.float32)
    ones = np.bincount(flat, weights=np.broadcast_to(sy, (h, k)).ravel(), minlength=h * 8).reshape(h, 8).astype(np.float32)
    zeros = counts - ones
    arity = np.array([len(ix) for ix in subsets], dtype=np.float32)
    occupied = (counts > 0).sum(1)
    minority = np.minimum(zeros, ones).sum(1)
    p = ones / np.maximum(counts, 1)
    entropy = -(p * np.log2(np.maximum(p, 1e-8)) + (1 - p) * np.log2(np.maximum(1 - p, 1e-8)))
    feat = np.stack([minority / k,
                     (zeros * ones).sum(1) / np.maximum((counts * (counts - 1) / 2).sum(1), 1),
                     occupied / (2 ** arity), (counts == 1).sum(1) / np.maximum(occupied, 1),
                     (entropy * counts).sum(1) / k, arity / 3,
                     np.log1p(k / (2 ** arity)) / np.log(65),
                     np.full(h, abs(float(sy.mean()) - .5) * 2),
                     np.full(h, np.log1p(k) / np.log(65)), occupied / k], axis=-1).astype(np.float32)
    qb = (qx @ codes).T
    qc = np.take_along_axis(counts, qb, axis=1)
    qo = np.take_along_axis(ones, qb, axis=1)
    # Jeffreys smoothing is a supplied statistical interface, not a neural output.
    values = ((qo + .5) / (qc + 1)).astype(np.float32)
    consistent = minority == 0
    lower = np.where(qc > 0, qo / np.maximum(qc, 1), 0)
    upper = np.where(qc > 0, qo / np.maximum(qc, 1), 1)
    if consistent.any():
        lo = lower[consistent].min(0); hi = upper[consistent].max(0)
        forced = lo == hi
    else:
        lo = np.zeros(len(qx)); hi = np.ones(len(qx)); forced = np.zeros(len(qx), dtype=bool)
    return dict(features=feat, values=values, subsets=subsets, consistent=consistent, arity=arity,
                lower=lo, upper=hi, forced=forced, raw_values=(lower + upper) / 2,
                query_seen=qc > 0)


def symbolic_prediction(encoded):
    """Written Occam baseline: average the smallest support-consistent subsets.

    If observations contradict every hypothesis, choose minimum empirical error
    and then minimum arity. This is a baseline, never a neural fallback.
    """
    errors = encoded['features'][:, 0]
    valid = errors == errors.min()
    valid &= encoded['arity'] == encoded['arity'][valid].min()
    return encoded['raw_values'][valid].mean(0)


def nearest_neighbor(support_x, support_y, query_x):
    sx, qx = np.asarray(support_x), np.asarray(query_x)
    distance = (qx[:, None, :] != sx[None, :, :]).sum(-1)
    nearest = distance == distance.min(1, keepdims=True)
    return (nearest * np.asarray(support_y)[None, :]).sum(1) / nearest.sum(1)
