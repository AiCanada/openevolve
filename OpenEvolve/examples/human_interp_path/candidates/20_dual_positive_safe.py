"""Conservative dual-positive recipe: prioritize min(evolve,holdout) generalization."""
import numpy as np
# EVOLVE-BLOCK-START


def _kabsch_R(P, Q):
    if P.shape[0] < 3:
        return np.eye(3)
    mu_p, mu_q = P.mean(0), Q.mean(0)
    H = (P - mu_p).T @ (Q - mu_q)
    try:
        U, _S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return np.eye(3)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def _so3_log(R):
    cos_t = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_t))
    if theta < 1e-8:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (theta / (2.0 * np.sin(theta) + 1e-12))


def _so3_exp(w):
    theta = float(np.linalg.norm(w))
    if theta < 1e-10:
        return np.eye(3)
    k = w / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def predict_interior(ca_start, ca_end, fracs, mask):
    """v3 dual-positive-ish: ~45% screw, 12% local, rest linear (safer holdout)."""
    ca0 = np.asarray(ca_start, float)
    ca1 = np.asarray(ca_end, float)
    m = np.asarray(mask, bool)
    fracs = np.asarray(fracs, float)
    L, Ti = ca0.shape[0], len(fracs)
    half = 13

    lin = np.stack([(1 - s) * ca0 + s * ca1 for s in fracs], 0)

    if m.sum() >= 3:
        R = _kabsch_R(ca0[m], ca1[m])
        mu0, mu1 = ca0[m].mean(0), ca1[m].mean(0)
        om = _so3_log(R)
        res = ca1 - ((ca0 - mu0) @ R.T + mu1)
        screw = np.empty((Ti, L, 3))
        for i, s in enumerate(fracs):
            s = float(s)
            Rs = _so3_exp(om * s)
            mus = (1 - s) * mu0 + s * mu1
            screw[i] = (ca0 - mu0) @ Rs.T + mus + s * res
    else:
        screw = lin.copy()

    loc = np.empty((Ti, L, 3))
    for i in range(L):
        lo, hi = max(0, i - half), min(L, i + half + 1)
        sel = np.arange(lo, hi)
        sel = sel[m[sel]]
        if sel.size < 3:
            loc[:, i] = lin[:, i]
            continue
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0, mu1 = ca0[sel].mean(0), ca1[sel].mean(0)
        om = _so3_log(R)
        res = ca1[i] - ((ca0[i] - mu0) @ R.T + mu1)
        for ti, s in enumerate(fracs):
            s = float(s)
            Rs = _so3_exp(om * s)
            mus = (1 - s) * mu0 + s * mu1
            loc[ti, i] = (ca0[i] - mu0) @ Rs.T + mus + s * res

    # from dual-positive log: lower screw than champion, more lin
    return 0.43 * lin + 0.45 * screw + 0.12 * loc
# EVOLVE-BLOCK-END


def run_interpolation(ca_start, ca_end, fracs, mask):
    return predict_interior(ca_start, ca_end, fracs, mask)
