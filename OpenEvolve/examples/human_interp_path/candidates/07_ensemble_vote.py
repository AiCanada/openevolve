"""Transition-path interpolation for protein backbones (Project H.U.M.A.N.)."""
import numpy as np

# EVOLVE-BLOCK-START


def _kabsch_R(P, Q, w=None):
    if w is None:
        w = np.ones(P.shape[0], dtype=np.float64)
    w = w.astype(np.float64)
    sw = w.sum()
    if sw < 1e-12 or P.shape[0] < 3:
        return np.eye(3)
    mu_p = (w[:, None] * P).sum(0) / sw
    mu_q = (w[:, None] * Q).sum(0) / sw
    X = (P - mu_p) * w[:, None]
    Y = Q - mu_q
    H = X.T @ Y
    try:
        U, _S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return np.eye(3)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    if d == 0:
        d = 1.0
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def _so3_log(R):
    cos_t = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_t))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if abs(np.pi - theta) < 1e-6:
        axis = np.sqrt(np.maximum(np.diag(R) + 1.0, 0.0))
        if axis.sum() < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            if R[2, 1] < R[1, 2]:
                axis[0] = -axis[0]
            if R[0, 2] < R[2, 0]:
                axis[1] = -axis[1]
            if R[1, 0] < R[0, 1]:
                axis[2] = -axis[2]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    return w * (theta / (2.0 * np.sin(theta) + 1e-12))


def _so3_exp(w):
    theta = float(np.linalg.norm(w))
    if theta < 1e-10:
        return np.eye(3)
    k = w / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _restore_ca_bonds(coords, mask, target=3.8, strength=0.35):
    """Pull consecutive Cα spacing toward ~3.8 A (gentle, keeps overall shape)."""
    x = coords.copy()
    idx = np.where(mask)[0]
    if idx.size < 2:
        return x
    for _ in range(2):
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:
                continue  # only sequential chain neighbors
            d = x[b] - x[a]
            n = float(np.linalg.norm(d))
            if n < 1e-8:
                continue
            corr = (target - n) * strength * 0.5
            u = d / n
            x[a] = x[a] - corr * u
            x[b] = x[b] + corr * u
    return x

def predict_interior(ca_start, ca_end, fracs, mask):
    """Average linear, global screw+res, and 3-seg screw (ensemble)."""
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    L = ca0.shape[0]
    Ti = len(fracs)
    lin = np.empty((Ti, L, 3), dtype=np.float64)
    for i, s in enumerate(fracs):
        lin[i] = (1.0 - s) * ca0 + s * ca1
    if m.sum() < 3:
        return lin
    w = m.astype(np.float64)
    sw = w.sum()
    mu0 = (w[:, None] * ca0).sum(0) / sw
    mu1 = (w[:, None] * ca1).sum(0) / sw
    R = _kabsch_R(ca0[m], ca1[m])
    omega = _so3_log(R)
    residual = ca1 - ((ca0 - mu0) @ R.T + mu1)
    screw = np.empty_like(lin)
    for i, s in enumerate(fracs):
        s = float(s)
        Rs = _so3_exp(omega * s)
        mus = (1.0 - s) * mu0 + s * mu1
        screw[i] = (ca0 - mu0) @ Rs.T + mus + s * residual
    # 3-seg
    idx = np.where(m)[0]
    nseg = 3
    cuts = np.linspace(0, idx.size, nseg + 1).astype(int)
    multi = lin.copy()
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 3:
            continue
        sel = idx[a:b]
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0s = ca0[sel].mean(0)
        mu1s = ca1[sel].mean(0)
        om = _so3_log(R)
        res = ca1[sel] - ((ca0[sel] - mu0s) @ R.T + mu1s)
        for i, s in enumerate(fracs):
            s = float(s)
            Rs = _so3_exp(om * s)
            mus = (1.0 - s) * mu0s + s * mu1s
            multi[i, sel] = (ca0[sel] - mu0s) @ Rs.T + mus + s * res
    # weighted: favor multi/screw slightly over pure linear
    return 0.25 * lin + 0.35 * screw + 0.40 * multi

# EVOLVE-BLOCK-END


def run_interpolation(ca_start, ca_end, fracs, mask):
    return predict_interior(ca_start, ca_end, fracs, mask)
