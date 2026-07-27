"""Convex mix of v4 balanced-champ and dual_min recipes (Pareto middle)."""
import numpy as np
# EVOLVE-BLOCK-START

# alpha=0 -> dual_min, alpha=1 -> balanced high-evolve
ALPHA = 0.8


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


def _path(ca0, ca1, m, fracs, half, w_lin, w_screw, w_local, w_mode, mode_mid, s_gate, half_s, w_ms_s):
    L, Ti = ca0.shape[0], len(fracs)
    lin = np.stack([(1 - s) * ca0 + s * ca1 for s in fracs], 0)
    if m.sum() < 3:
        return lin
    Rg = _kabsch_R(ca0[m], ca1[m])
    mu0, mu1 = ca0[m].mean(0), ca1[m].mean(0)
    omg = _so3_log(Rg)
    res_g = ca1 - ((ca0 - mu0) @ Rg.T + mu1)
    screw = np.empty((Ti, L, 3))
    for i, s in enumerate(fracs):
        s = float(s)
        Rs = _so3_exp(omg * s)
        mus = (1 - s) * mu0 + s * mu1
        screw[i] = (ca0 - mu0) @ Rs.T + mus + s * res_g

    def local_scale(half_w):
        loc = np.empty((Ti, L, 3))
        for i in range(L):
            lo, hi = max(0, i - half_w), min(L, i + half_w + 1)
            sel = np.arange(lo, hi)
            sel = sel[m[sel]]
            if sel.size < 3:
                loc[:, i] = lin[:, i]
                continue
            Rl = _kabsch_R(ca0[sel], ca1[sel])
            m0, m1 = ca0[sel].mean(0), ca1[sel].mean(0)
            oml = _so3_log(Rl)
            resi = ca1[i] - ((ca0[i] - m0) @ Rl.T + m1)
            for ti, s in enumerate(fracs):
                s = float(s)
                Rs = _so3_exp(oml * s)
                mus = (1 - s) * m0 + s * m1
                loc[ti, i] = (ca0[i] - m0) @ Rs.T + mus + s * resi
        return loc

    loc_m = local_scale(half)
    if w_ms_s > 0.02:
        loc = (1 - w_ms_s) * loc_m + w_ms_s * local_scale(half_s)
    else:
        loc = loc_m

    mode_path = lin.copy()
    if w_mode > 0.02 and m.sum() >= 6:
        idxm = np.where(m)[0]
        D = ca1[idxm] - ca0[idxm]
        try:
            U, S, Vt = np.linalg.svd(D, full_matrices=False)
            k = min(1, U.shape[1], 3)
            recon = np.outer(U[:, 0] * S[0], Vt[0]) if k else np.zeros_like(D)
            miss = D - recon
            for i, s in enumerate(fracs):
                s = float(s)
                mid = mode_mid * np.sin(np.pi * s)
                delta = np.zeros((L, 3))
                delta[idxm] = (s + mid) * recon + s * miss
                mode_path[i] = ca0 + delta
        except np.linalg.LinAlgError:
            pass

    raw = np.array([w_lin, w_screw, w_local, w_mode], float)
    raw = np.maximum(raw, 0)
    raw = raw / (raw.sum() + 1e-12)
    out = np.empty((Ti, L, 3))
    for i, s in enumerate(fracs):
        s = float(s)
        mid = np.sin(np.pi * s)
        w = raw.copy()
        take = min(w[0], s_gate * mid * 0.25)
        w[0] -= take
        w[1] += take * 0.65
        w[2] += take * 0.35
        w = w / (w.sum() + 1e-12)
        out[i] = w[0] * lin[i] + w[1] * screw[i] + w[2] * loc[i] + w[3] * mode_path[i]
    return out


def predict_interior(ca_start, ca_end, fracs, mask):
    ca0 = np.asarray(ca_start, float)
    ca1 = np.asarray(ca_end, float)
    m = np.asarray(mask, bool)
    fracs = np.asarray(fracs, float)
    # balanced champ
    bal = _path(ca0, ca1, m, fracs, half=13, w_lin=0.122, w_screw=0.675, w_local=0.145,
                w_mode=0.057, mode_mid=0.113, s_gate=0.0, half_s=5, w_ms_s=0.0)
    # dual min
    dual = _path(ca0, ca1, m, fracs, half=20, w_lin=0.107, w_screw=0.661, w_local=0.175,
                 w_mode=0.056, mode_mid=0.113, s_gate=0.147, half_s=3, w_ms_s=0.134)
    a = float(ALPHA)
    return a * bal + (1 - a) * dual
# EVOLVE-BLOCK-END


def run_interpolation(ca_start, ca_end, fracs, mask):
    return predict_interior(ca_start, ca_end, fracs, mask)
