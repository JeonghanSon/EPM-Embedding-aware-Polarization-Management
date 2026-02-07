# src/delta/delta.py
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _dedup_keep_last(edge_list):
    """
    Undirected keep-last snapshot using (min, max) as the key.
    Removes self-loops and zero-weight edges.
    """
    last = {}
    for u, v, w in edge_list:
        u = int(u)
        v = int(v)
        if u == v or w == 0:
            continue
        a, b = (u, v) if u < v else (v, u)
        last[(a, b)] = float(w)  # file order = last
    return [(a, b, w) for (a, b), w in last.items()]


def _scaled_edges(edge_list, pos=1.0, neg=0.1):
    """
    Use sign only: + -> pos, - -> neg
    """
    out = []
    for u, v, w in edge_list:
        out.append((int(u), int(v), float(pos if w > 0 else neg)))
    return out


def delta_pinv_multi(L_pinv: np.ndarray, X) -> float:
    """
    Axis-wise delta from precomputed L^dagger:

        delta = sqrt( (1/k) * sum_{t=1..k} x_t^T L^dagger x_t )

    L_pinv: [N,N] (pseudo-inverse of Laplacian)
    X: [N,k] opinion coordinates (columns = axes)
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array [N,k], got shape={X.shape}")

    N, k = X.shape
    if k <= 0:
        return 0.0

    L_pinv = np.asarray(L_pinv, dtype=np.float64)
    if L_pinv.shape != (N, N):
        raise ValueError(f"L_pinv shape must be {(N, N)}, got {L_pinv.shape}")

    s = 0.0
    for t in range(k):
        x = X[:, t]
        x = x - x.mean()
        q = float(x.T @ (L_pinv @ x))
        if q < 0 and q > -1e-9:
            q = 0.0
        s += q

    m = s / float(k)
    if m < 0 and m > -1e-9:
        m = 0.0
    return float(np.sqrt(max(m, 0.0)))


def delta_solver_multi(
    edge_list,
    X,
    pos=1.0,
    neg=0.1,
    rtol=1e-5,
    atol=1e-8,
    maxiter=5000,
):
    """
    Axis-wise delta via a Laplacian solver (large graphs).
    Uses a keep-last undirected snapshot + sign scaling.

        delta = sqrt( (1/k) * sum_t x_t^T L^dagger x_t )

    Returns NaN on solver failure.
    """
    try:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            return float("nan")
        N, k = X.shape
        if k <= 0:
            return 0.0

        edges = _scaled_edges(_dedup_keep_last(edge_list), pos=pos, neg=neg)

        rows, cols, data = [], [], []
        for u, v, w in edges:
            if 0 <= u < N and 0 <= v < N and u != v:
                rows += [u, v]
                cols += [v, u]
                data += [w, w]
        A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        deg = np.asarray(A.sum(axis=1)).ravel()
        L = (sp.diags(deg) - A).tocsr()

        ones = np.ones(N, dtype=np.float64)

        # Solve (L + (1/n)11^T) y = b to get y = L^dagger b (for centered b).
        def matvec(x):
            return (L @ x) + (x.mean() * ones)

        Aop = spla.LinearOperator((N, N), matvec=matvec, dtype=np.float64)

        inv_diag = 1.0 / np.maximum(L.diagonal().astype(np.float64), 1e-12)
        Mop = spla.LinearOperator((N, N), matvec=lambda x: inv_diag * x, dtype=np.float64)

        s = 0.0
        for t in range(k):
            b = X[:, t]
            # In the solver path, centering is enforced (definition consistency + numerical stability).
            b = b - b.mean()

            y, info = spla.cg(Aop, b, M=Mop, rtol=rtol, atol=atol, maxiter=maxiter)
            if info != 0:
                return float("nan")
            y = y - y.mean()

            q = float(b.T @ y)  # b^T L^dagger b
            if q < 0 and q > -1e-9:
                q = 0.0
            s += q

        m = s / float(k)
        if m < 0 and m > -1e-9:
            m = 0.0
        return float(np.sqrt(max(m, 0.0)))
    except Exception:
        return float("nan")
