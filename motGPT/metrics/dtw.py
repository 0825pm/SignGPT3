"""
Dynamic Time Warping (DTW) for Sign Language Evaluation
Based on SOKE implementation
"""

from numpy import array, zeros, full, argmin, inf
import numpy as np
from math import isinf


def rigid_align(source, target):
    """
    Procrustes alignment: align source to target using SVD.
    
    Args:
        source: (N, 3) source points
        target: (N, 3) target points
    
    Returns:
        aligned: (N, 3) aligned source points
    """
    # Center the points
    mu_source = source.mean(axis=0, keepdims=True)
    mu_target = target.mean(axis=0, keepdims=True)
    
    source_centered = source - mu_source
    target_centered = target - mu_target
    
    # Compute optimal rotation using SVD
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    
    R = Vt.T @ U.T
    
    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute scale
    var_source = (source_centered ** 2).sum()
    scale = S.sum() / max(var_source, 1e-8)
    
    # Apply transformation
    aligned = scale * (source_centered @ R.T) + mu_target
    
    return aligned


def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    Args:
        x: N1 x M array (sequence 1)
        y: N2 x M array (sequence 2)
        dist: distance function used as cost measure
        warp: how many shifts are computed
        w: window size limiting the maximal distance between indices
        s: weight applied on off-diagonal moves
    
    Returns:
        Tuple of (minimum distance, cost matrix, accumulated cost matrix, wrap path)
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    
    r, c = len(x), len(y)
    
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    
    D1 = D0[1:, 1:]  # view
    
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    
    C = D1.copy()
    jrange = range(c)
    
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    
    return D1[-1, -1], C, D1, path


def _traceback(D):
    """Traceback to find optimal path"""
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)


def l2_dist_align(x, y, wanted=None, align_idx=None):
    """
    Compute L2 distance with optional alignment.
    
    Args:
        x: (N, 3) predicted joints for one frame
        y: (N, 3) ground truth joints for one frame
        wanted: list of joint indices to use (None = all)
        align_idx: joint index for translation alignment
                   - None: use Procrustes (rigid_align) → DTW-PA-JPE
                   - 0: use wrist alignment → DTW-JPE
    
    Returns:
        dist: mean L2 distance
    """
    if align_idx is None:
        # DTW-PA-JPE: Procrustes alignment (translation + rotation + scale)
        x = rigid_align(x, y)
    else:
        # DTW-JPE: Translation alignment only (wrist-relative)
        x = x - x[align_idx:align_idx+1] + y[align_idx:align_idx+1]
    
    if wanted is not None:
        x = x[wanted]
        y = y[wanted]
    
    dist = np.mean(np.sqrt(((x - y) ** 2).sum(axis=1)))
    return dist


def l2_dist(x, y, wanted=None):
    """Simple L2 distance without alignment"""
    if wanted is not None:
        x = x[wanted]
        y = y[wanted]
    dist = np.mean(np.sqrt(((x - y) ** 2).sum(axis=1)))
    return dist