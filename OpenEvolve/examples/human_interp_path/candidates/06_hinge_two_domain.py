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
    """Split chain at residue of max bending; two-domain screws + blend."""
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    L = ca0.shape[0]
    Ti = len(fracs)
    out = np.empty((Ti, L, 3), dtype=np.float64)
    idx = np.where(m)[0]
    if idx.size < 10:
        for i, s in enumerate(fracs):
            out[i] = (1.0 - s) * ca0 + s * ca1
        return out
    # hinge = argmax local displacement gradient along chain
    d = np.linalg.norm(ca1 - ca0, axis=1)
    grad = np.zeros(L)
    for a, b in zip(idx[:-1], idx[1:]):
        if b == a + 1:
            grad[b] = abs(d[b] - d[a])
    hinge = int(np.argmax(grad)) if grad.max() > 0 else int(idx[idx.size // 2])
    # ensure hinge interior
    hinge = int(np.clip(hinge, idx[3], idx[-4]))
    left = idx[idx < hinge]
    right = idx[idx >= hinge]
    domains = []
    for sel in (left, right):
        if sel.size < 3:
            continue
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0 = ca0[sel].mean(0)
        mu1 = ca1[sel].mean(0)
        omega = _so3_log(R)
        residual = ca1[sel] - ((ca0[sel] - mu0) @ R.T + mu1)
        domains.append((sel, mu0, mu1, omega, residual))
    for i, s in enumerate(fracs):
        s = float(s)
        frame = (1.0 - s) * ca0 + s * ca1
        for sel, mu0, mu1, omega, residual in domains:
            Rs = _so3_exp(omega * s)
            mus = (1.0 - s) * mu0 + s * mu1
            frame[sel] = (ca0[sel] - mu0) @ Rs.T + mus + s * residual
        # smooth near hinge
        for j in range(max(0, hinge - 2), min(L, hinge + 3)):
            if not m[j]:
                continue
            frame[j] = 0.5 * frame[j] + 0.5 * ((1.0 - s) * ca0[j] + s * ca1[j])
        out[i] = frame
    return out

# EVOLVE-BLOCK-END


def run_interpolation(ca_start, ca_end, fracs, mask):
    return predict_interior(ca_start, ca_end, fracs, mask)
