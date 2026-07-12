"""
G4Hunter (Bedrat et al. 2016) per-base G-quadruplex propensity track.

Runs of G score +min(run,4) per base; runs of C score -min(run,4); A/T score 0.
The raw scores are smoothed over a sliding window and scaled to ~[-1, 1], giving
a 1 x L track that can be appended as an extra input channel next to the 4 x L
one-hot. Positive = G4-forming (G-rich) potential; negative = C-rich (i-motif
complementary strand) potential. Deterministic — no training/data needed.
"""
from __future__ import annotations
import numpy as np
from common import clean_seq


def g4hunter_track(seq: str, window: int = 25) -> np.ndarray:
    s = clean_seq(seq)
    L = len(s)
    raw = np.zeros(L, dtype=np.float32)
    i = 0
    while i < L:
        c = s[i]
        if c in "GC":
            j = i
            while j < L and s[j] == c:
                j += 1
            run = min(j - i, 4)
            raw[i:j] = run if c == "G" else -run
            i = j
        else:
            i += 1
    if window > 1 and L > 0:
        k = np.ones(min(window, L), dtype=np.float32)
        raw = np.convolve(raw, k / k.size, mode="same")
    return (raw / 4.0).astype(np.float32)          # scale to ~[-1, 1]


def g4_channel_batch(seqs, length: int) -> np.ndarray:
    """Return (N, 1, length) G4 track, centre-cropped/padded to match one_hot."""
    from common import clean_seq
    out = np.zeros((len(seqs), 1, length), dtype=np.float32)
    for n, sq in enumerate(seqs):
        s = clean_seq(sq)
        if len(s) > length:
            st = (len(s) - length) // 2
            s = s[st:st + length]
        t = g4hunter_track(s)
        if len(t) < length:
            pad = length - len(t); left = pad // 2
            t = np.concatenate([np.zeros(left, np.float32), t, np.zeros(pad - left, np.float32)])
        out[n, 0] = t[:length]
    return out


def mean_g4(seq: str) -> float:
    """Sequence-level mean G4Hunter score (a scalar propensity)."""
    t = g4hunter_track(seq)
    return float(np.mean(t)) if len(t) else 0.0


def imotif_track(seq: str, window: int = 25) -> np.ndarray:
    """i-motif (C-rich) propensity — G4Hunter with the sign flipped so C-runs
    are positive. NOTE: this is exactly -G4Hunter, i.e. the same C/G-run signal
    the G4 channel already carries as its negative half; provided for explicit
    ablation. Antisymmetric under reverse-complement (C-runs <-> G-runs)."""
    return -g4hunter_track(seq, window)


def rloop_track(seq: str, window: int = 50) -> np.ndarray:
    """R-loop-forming propensity proxy: windowed GC skew = smoothed (nG - nC) /
    (nG + nC + eps). R-loops favour sustained G-rich / GC-skewed stretches on one
    strand over ~hundreds of bp, so this uses a wider window and flat G=+1 / C=-1
    weighting (distinct from G4Hunter's local run-length score). Antisymmetric
    under reverse-complement."""
    s = clean_seq(seq)
    g = np.array([1.0 if b == "G" else -1.0 if b == "C" else 0.0 for b in s], np.float32)
    if window > 1 and len(s):
        k = np.ones(min(window, len(s)), np.float32)
        num = np.convolve(g, k, mode="same")
        den = np.convolve(np.abs(g), k, mode="same") + 1e-6
        return (num / den).astype(np.float32)
    return g


_TRACKS = {"g4": g4hunter_track, "imotif": imotif_track, "rloop": rloop_track}


def build_nonb_channels(seqs, length, feats):
    """Return (N, len(feats), length) stacked non-B channels, centre-cropped/
    padded like one_hot. feats: subset of ['g4','imotif','rloop'] in order."""
    out = np.zeros((len(seqs), len(feats), length), np.float32)
    for n, sq in enumerate(seqs):
        s = clean_seq(sq)
        if len(s) > length:
            st = (len(s) - length) // 2
            s = s[st:st + length]
        for fi, f in enumerate(feats):
            t = _TRACKS[f](s)
            if len(t) < length:
                pad = length - len(t); left = pad // 2
                t = np.concatenate([np.zeros(left, np.float32), t, np.zeros(pad - left, np.float32)])
            out[n, fi] = t[:length]
    return out
