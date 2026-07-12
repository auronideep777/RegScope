"""
Shared utilities: DNA encoding, chromosome-aware splitting, metrics.
Torch-free so baselines / data-prep can import without the DL stack.
"""
from __future__ import annotations
import re
import numpy as np
from itertools import product

BASES = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}
COMPLEMENT = str.maketrans("ACGTacgtN", "TGCAtgcaN")


def clean_seq(seq: str) -> str:
    return re.sub(r"[^ACGTN]", "N", str(seq).upper())


def reverse_complement(seq: str) -> str:
    return clean_seq(seq).translate(COMPLEMENT)[::-1]


def one_hot(seq: str, length: int | None = None) -> np.ndarray:
    seq = clean_seq(seq)
    if length is not None:
        if len(seq) > length:
            start = (len(seq) - length) // 2
            seq = seq[start:start + length]
        elif len(seq) < length:
            pad = length - len(seq)
            left = pad // 2
            seq = "N" * left + seq + "N" * (pad - left)
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq):
        j = BASE_TO_IDX.get(base, -1)
        if j >= 0:
            arr[j, i] = 1.0
    return arr


def batch_one_hot(seqs, length: int) -> np.ndarray:
    return np.stack([one_hot(s, length) for s in seqs]).astype(np.float32)


def rc_onehot(x: np.ndarray) -> np.ndarray:
    """Reverse-complement a one-hot tensor (..., 4, L): flip ACGT<->TGCA and reverse L."""
    return x[..., ::-1, ::-1].copy()


# --------------------------------------------------------------------------- #
def chrom_split(chroms, val_chroms=("chr8", "chr9"),
                test_chroms=("chr7", "chr17")):
    chroms = np.asarray([str(c) for c in chroms])
    is_val = np.isin(chroms, list(val_chroms))
    is_test = np.isin(chroms, list(test_chroms))
    return ~(is_val | is_test), is_val, is_test


# --------------------------------------------------------------------------- #
def spearman(y_true, y_pred):
    from scipy.stats import spearmanr
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("nan")
    return float(spearmanr(y_true[m], y_pred[m]).correlation)


def pearson(y_true, y_pred):
    from scipy.stats import pearsonr
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("nan")
    return float(pearsonr(y_true[m], y_pred[m])[0])


def r2(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("nan")
    yt, yp = y_true[m], y_pred[m]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def auroc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    y_true, y_score = np.asarray(y_true, float), np.asarray(y_score, float)
    m = np.isfinite(y_true) & np.isfinite(y_score)
    if m.sum() < 3 or len(np.unique(y_true[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y_true[m], y_score[m]))


def sign_accuracy(observed, predicted, min_abs=0.0):
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    m = np.isfinite(observed) & np.isfinite(predicted) & (np.abs(observed) > min_abs)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(observed[m]) == np.sign(predicted[m])))
