"""Multi-scale local screw ensemble: blend half-windows 5 / 12 / 22 + global."""
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


def _local_path(ca0, ca1, m, fracs, half):
    L, Ti = ca0.shape[0], len(fracs)
    out = np.empty((Ti, L, 3))
    for i in range(L):
        lo, hi = max(0, i - half), min(L, i + half + 1)
        sel = np.arange(lo, hi)
        sel = sel[m[sel]]
        if sel.size < 3:
            for t, s in enumerate(fracs):
                out[t, i] = (1 - s) * ca0[i] + s * ca1[i]
            continue
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0, mu1 = ca0[sel].mean(0), ca1[sel].mean(0)
        om = _so3_log(R)
        res = ca1[i] - ((ca0[i] - mu0) @ R.T + mu1)
        for ti, s in enumerate(fracs):
            s = float(s)
            Rs = _so3_exp(om * s)
            mus = (1 - s) * mu0 + s * mu1
            out[ti, i] = (ca0[i] - mu0) @ Rs.T + mus + s * res
    return out


def predict_interior(ca_start, ca_end, fracs, mask):
    ca0 = np.asarray(ca_start, float)
    ca1 = np.asarray(ca_end, float)
    m = np.asarray(mask, bool)
    fracs = np.asarray(fracs, float)
    L = ca0.shape[0]
    Ti = len(fracs)

    # global screw+residual
    lin = np.empty((Ti, L, 3))
    for i, s in enumerate(fracs):
        lin[i] = (1 - s) * ca0 + s * ca1
    if m.sum() >= 3:
        R = _kabsch_R(ca0[m], ca1[m])
        mu0, mu1 = ca0[m].mean(0), ca1[m].mean(0)
        om = _so3_log(R)
        res = ca1 - ((ca0 - mu0) @ R.T + mu1)
        glob = np.empty((Ti, L, 3))
        for i, s in enumerate(fracs):
            s = float(s)
            Rs = _so3_exp(om * s)
            mus = (1 - s) * mu0 + s * mu1
            glob[i] = (ca0 - mu0) @ Rs.T + mus + s * res
    else:
        glob = lin.copy()

    loc_s = _local_path(ca0, ca1, m, fracs, 5)
    loc_m = _local_path(ca0, ca1, m, fracs, 12)
    loc_l = _local_path(ca0, ca1, m, fracs, 22)

    # weights tuned toward generalization: global-heavy + light multiscale
    w_g, w_s, w_m, w_l = 0.55, 0.10, 0.20, 0.15
    return w_g * glob + w_s * loc_s + w_m * loc_m + w_l * loc_l
# EVOLVE-BLOCK-END


def run_interpolation(ca_start, ca_end, fracs, mask):
    return predict_interior(ca_start, ca_end, fracs, mask)
