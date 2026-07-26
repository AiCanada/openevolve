"""Materialized v3 champion params: screw+local+mode dual-positive blend."""
import numpy as np
# EVOLVE-BLOCK-START

PARAMS = {
    "nseg": 1,
    "n_hinge": 3,
    "n_modes": 1,
    "ease": 0.0,
    "hermite": 0.0,
    "w_lin": 0.122,
    "w_screw": 0.634,
    "w_multi": 0.041,
    "w_local": 0.145,
    "w_hinge": 0.0,
    "w_mode": 0.057,
    "bond_strength": 0.0,
    "local_half": 13,
    "soft_local": 0.0,
    "res_mid_boost": 0.0,
    "mode_mid_boost": 0.113,
}


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


def _time(s, ease, hermite):
    s = float(s)
    h = float(np.clip(hermite, 0.0, 1.0))
    e = float(np.clip(ease, 0.0, 1.0))
    sh = 3 * s * s - 2 * s * s * s
    se = s * s * (3.0 - 2.0 * s)
    t = (1.0 - h) * ((1.0 - e) * s + e * se) + h * sh
    return float(np.clip(t, 0.0, 1.0))


def predict_interior(ca_start, ca_end, fracs, mask):
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    fracs = np.asarray(fracs, dtype=np.float64)
    p = PARAMS
    L = ca0.shape[0]
    Ti = len(fracs)
    ease = float(p.get("ease", 0.0))
    herm = float(p.get("hermite", 0.0))
    half = max(2, int(p.get("local_half", 13)))
    nseg = max(1, int(p.get("nseg", 3)))
    nmodes = max(0, int(p.get("n_modes", 1)))

    lin = np.empty((Ti, L, 3))
    for i, s in enumerate(fracs):
        t = _time(s, ease, herm)
        lin[i] = (1.0 - t) * ca0 + t * ca1

    screw = np.empty((Ti, L, 3))
    if m.sum() >= 3:
        w = m.astype(np.float64)
        sw = w.sum()
        mu0 = (w[:, None] * ca0).sum(0) / sw
        mu1 = (w[:, None] * ca1).sum(0) / sw
        R = _kabsch_R(ca0[m], ca1[m])
        omega = _so3_log(R)
        residual = ca1 - ((ca0 - mu0) @ R.T + mu1)
        for i, s in enumerate(fracs):
            t = _time(s, ease, herm)
            Rs = _so3_exp(omega * t)
            mus = (1.0 - t) * mu0 + t * mu1
            screw[i] = (ca0 - mu0) @ Rs.T + mus + t * residual
    else:
        screw = lin.copy()

    multi = screw.copy()
    idx = np.where(m)[0]
    if idx.size >= nseg * 3 and nseg > 1:
        cuts = np.linspace(0, idx.size, nseg + 1).astype(int)
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < 3:
                continue
            sel = idx[a:b]
            R = _kabsch_R(ca0[sel], ca1[sel])
            mu0s, mu1s = ca0[sel].mean(0), ca1[sel].mean(0)
            om = _so3_log(R)
            res = ca1[sel] - ((ca0[sel] - mu0s) @ R.T + mu1s)
            for i, s in enumerate(fracs):
                t = _time(s, ease, herm)
                Rs = _so3_exp(om * t)
                mus = (1.0 - t) * mu0s + t * mu1s
                multi[i, sel] = (ca0[sel] - mu0s) @ Rs.T + mus + t * res

    loc = np.empty((Ti, L, 3))
    for i in range(L):
        lo, hi = max(0, i - half), min(L, i + half + 1)
        sel = np.arange(lo, hi)
        sel = sel[m[sel]]
        if sel.size < 3:
            loc[:, i] = lin[:, i]
            continue
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0s, mu1s = ca0[sel].mean(0), ca1[sel].mean(0)
        om = _so3_log(R)
        res = ca1[i] - ((ca0[i] - mu0s) @ R.T + mu1s)
        for ti, s in enumerate(fracs):
            t = _time(s, ease, herm)
            Rs = _so3_exp(om * t)
            mus = (1.0 - t) * mu0s + t * mu1s
            loc[ti, i] = (ca0[i] - mu0s) @ Rs.T + mus + t * res

    mode_path = lin.copy()
    if nmodes > 0 and m.sum() >= 6:
        idxm = np.where(m)[0]
        D = ca1[idxm] - ca0[idxm]
        try:
            U, S, Vt = np.linalg.svd(D, full_matrices=False)
            k = min(nmodes, U.shape[1], 3)
            recon = np.zeros_like(D)
            for j in range(k):
                recon += np.outer(U[:, j] * S[j], Vt[j])
            miss = D - recon
            beta = float(p.get("mode_mid_boost", 0.0))
            for i, s in enumerate(fracs):
                t = _time(s, ease, herm)
                mid = beta * np.sin(np.pi * s)
                delta = np.zeros((L, 3))
                delta[idxm] = (t + mid) * recon + t * miss
                mode_path[i] = ca0 + delta
        except np.linalg.LinAlgError:
            pass

    ws = np.array([
        float(p["w_lin"]), float(p["w_screw"]), float(p["w_multi"]),
        float(p["w_local"]), float(p["w_mode"]),
    ], dtype=np.float64)
    ws = np.maximum(ws, 0.0)
    ws = ws / (ws.sum() + 1e-12)
    return ws[0] * lin + ws[1] * screw + ws[2] * multi + ws[3] * loc + ws[4] * mode_path
# EVOLVE-BLOCK-END


def run_interpolation(ca_start, ca_end, fracs, mask):
    return predict_interior(ca_start, ca_end, fracs, mask)
