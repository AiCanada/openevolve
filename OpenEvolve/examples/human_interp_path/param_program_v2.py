"""Parameterized interpolator v2 — hermite timing + multi-component blend."""
import numpy as np

PARAMS = {
    "nseg": 3,
    "ease": 0.0,
    "hermite": 0.0,       # 0=linear s, 1=cubic hermite endpoint-velocity-0
    "w_lin": 0.5,
    "w_screw": 0.5,
    "w_multi": 0.0,
    "w_local": 0.0,
    "w_hinge": 0.0,
    "bond_strength": 0.0,
    "local_half": 8,
    "hinge_smooth": 2,
    "res_mid_boost": 0.0,  # anisotropic residual mid boost
    "mode_mid_boost": 0.0, # PCA mode mid sin boost
}


def inject_params(p: dict):
    PARAMS.update(p)


def _kabsch_R(P, Q):
    if P.shape[0] < 3:
        return np.eye(3)
    mu_p, mu_q = P.mean(0), Q.mean(0)
    H = (P - mu_p).T @ (Q - mu_q)
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
        return np.zeros(3)
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
        return axis / (np.linalg.norm(axis) + 1e-12) * theta
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
    # hermite zero-velocity endpoints: s' = 3s^2 - 2s^3
    sh = 3 * s * s - 2 * s * s * s
    # smoothstep
    se = s * s * (3.0 - 2.0 * s)
    # blend base s with ease and hermite
    t = (1.0 - h) * ((1.0 - e) * s + e * se) + h * sh
    return float(np.clip(t, 0.0, 1.0))


def _restore_bonds(x, mask, strength):
    if strength <= 1e-8:
        return x
    x = x.copy()
    idx = np.where(mask)[0]
    st = float(strength)
    for _ in range(2):
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:
                continue
            d = x[b] - x[a]
            n = float(np.linalg.norm(d))
            if n < 1e-8:
                continue
            corr = (3.8 - n) * st * 0.5
            u = d / n
            x[a] -= corr * u
            x[b] += corr * u
    return x


def _global_screw(ca0, ca1, m, fracs, ease, hermite, res_mid_boost):
    Ti, L = len(fracs), ca0.shape[0]
    out = np.empty((Ti, L, 3))
    if m.sum() < 3:
        for i, s in enumerate(fracs):
            t = _time(s, ease, hermite)
            out[i] = (1.0 - t) * ca0 + t * ca1
        return out
    w = m.astype(np.float64)
    sw = w.sum()
    mu0 = (w[:, None] * ca0).sum(0) / sw
    mu1 = (w[:, None] * ca1).sum(0) / sw
    R = _kabsch_R(ca0[m], ca1[m])
    omega = _so3_log(R)
    residual = ca1 - ((ca0 - mu0) @ R.T + mu1)
    rmag = np.linalg.norm(residual, axis=1)
    rmag = rmag / (rmag.max() + 1e-8)
    beta = float(res_mid_boost)
    for i, s in enumerate(fracs):
        s = float(s)
        t = _time(s, ease, hermite)
        Rs = _so3_exp(omega * t)
        mus = (1.0 - t) * mu0 + t * mu1
        rigid = (ca0 - mu0) @ Rs.T + mus
        prog = t + beta * np.sin(np.pi * s) * rmag
        out[i] = rigid + prog[:, None] * residual
    return out


def _multi(ca0, ca1, m, fracs, nseg, ease, hermite):
    base = _global_screw(ca0, ca1, m, fracs, ease, hermite, 0.0)
    idx = np.where(m)[0]
    nseg = max(1, int(nseg))
    if idx.size < nseg * 3:
        return base
    cuts = np.linspace(0, idx.size, nseg + 1).astype(int)
    out = base.copy()
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 3:
            continue
        sel = idx[a:b]
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0, mu1 = ca0[sel].mean(0), ca1[sel].mean(0)
        omega = _so3_log(R)
        residual = ca1[sel] - ((ca0[sel] - mu0) @ R.T + mu1)
        for i, s in enumerate(fracs):
            t = _time(s, ease, hermite)
            Rs = _so3_exp(omega * t)
            mus = (1.0 - t) * mu0 + t * mu1
            out[i, sel] = (ca0[sel] - mu0) @ Rs.T + mus + t * residual
    return out


def _local(ca0, ca1, m, fracs, half, ease, hermite):
    Ti, L = len(fracs), ca0.shape[0]
    out = np.empty((Ti, L, 3))
    half = max(2, int(half))
    for i in range(L):
        lo, hi = max(0, i - half), min(L, i + half + 1)
        sel = np.arange(lo, hi)
        sel = sel[m[sel]]
        if sel.size < 3:
            for ti, s in enumerate(fracs):
                t = _time(s, ease, hermite)
                out[ti, i] = (1.0 - t) * ca0[i] + t * ca1[i]
            continue
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0, mu1 = ca0[sel].mean(0), ca1[sel].mean(0)
        omega = _so3_log(R)
        res = ca1[i] - ((ca0[i] - mu0) @ R.T + mu1)
        for ti, s in enumerate(fracs):
            t = _time(s, ease, hermite)
            Rs = _so3_exp(omega * t)
            mus = (1.0 - t) * mu0 + t * mu1
            out[ti, i] = (ca0[i] - mu0) @ Rs.T + mus + t * res
    return out


def _hinge(ca0, ca1, m, fracs, smooth, ease, hermite):
    Ti, L = len(fracs), ca0.shape[0]
    idx = np.where(m)[0]
    if idx.size < 10:
        return _global_screw(ca0, ca1, m, fracs, ease, hermite, 0.0)
    d = np.linalg.norm(ca1 - ca0, axis=1)
    grad = np.zeros(L)
    for a, b in zip(idx[:-1], idx[1:]):
        if b == a + 1:
            grad[b] = abs(d[b] - d[a])
    hinge = int(np.argmax(grad)) if grad.max() > 0 else int(idx[idx.size // 2])
    hinge = int(np.clip(hinge, idx[3], idx[-4]))
    domains = []
    for sel in (idx[idx < hinge], idx[idx >= hinge]):
        if sel.size < 3:
            continue
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0, mu1 = ca0[sel].mean(0), ca1[sel].mean(0)
        omega = _so3_log(R)
        residual = ca1[sel] - ((ca0[sel] - mu0) @ R.T + mu1)
        domains.append((sel, mu0, mu1, omega, residual))
    out = np.empty((Ti, L, 3))
    sm = max(0, int(smooth))
    for i, s in enumerate(fracs):
        t = _time(s, ease, hermite)
        frame = (1.0 - t) * ca0 + t * ca1
        for sel, mu0, mu1, omega, residual in domains:
            Rs = _so3_exp(omega * t)
            mus = (1.0 - t) * mu0 + t * mu1
            frame[sel] = (ca0[sel] - mu0) @ Rs.T + mus + t * residual
        for j in range(max(0, hinge - sm), min(L, hinge + sm + 1)):
            if m[j]:
                frame[j] = 0.5 * frame[j] + 0.5 * ((1.0 - t) * ca0[j] + t * ca1[j])
        out[i] = frame
    return out


def _mode_path(ca0, ca1, m, fracs, ease, hermite, mid_boost):
    Ti, L = len(fracs), ca0.shape[0]
    out = np.empty((Ti, L, 3))
    d = (ca1 - ca0).reshape(-1)
    w = np.repeat(m.astype(np.float64), 3)
    d_w = d * w
    n = float(np.linalg.norm(d_w)) + 1e-12
    mode = d_w / n
    amp = float((d * w * mode).sum())
    beta = float(mid_boost)
    for i, s in enumerate(fracs):
        t = _time(s, ease, hermite)
        delta = (t * amp * mode + beta * np.sin(np.pi * s) * amp * mode).reshape(L, 3)
        full = t * (ca1 - ca0)
        miss = full - (t * amp * mode).reshape(L, 3)
        out[i] = ca0 + delta + miss
    return out


def predict_interior(ca_start, ca_end, fracs, mask):
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    fracs = np.asarray(fracs, dtype=np.float64)
    p = PARAMS
    ease = float(p.get("ease", 0.0))
    herm = float(p.get("hermite", 0.0))
    lin = np.stack([((1.0 - _time(s, ease, herm)) * ca0 + _time(s, ease, herm) * ca1) for s in fracs])
    screw = _global_screw(ca0, ca1, m, fracs, ease, herm, p.get("res_mid_boost", 0.0))
    multi = _multi(ca0, ca1, m, fracs, p.get("nseg", 3), ease, herm)
    loc = _local(ca0, ca1, m, fracs, p.get("local_half", 8), ease, herm)
    hin = _hinge(ca0, ca1, m, fracs, p.get("hinge_smooth", 2), ease, herm)
    mode = _mode_path(ca0, ca1, m, fracs, ease, herm, p.get("mode_mid_boost", 0.0))

    weights = np.array([
        float(p.get("w_lin", 0.5)),
        float(p.get("w_screw", 0.5)),
        float(p.get("w_multi", 0.0)),
        float(p.get("w_local", 0.0)),
        float(p.get("w_hinge", 0.0)),
        float(p.get("w_mode", 0.0)),
    ], dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    if weights.sum() < 1e-12:
        weights[0] = 1.0
    weights = weights / weights.sum()
    parts = [lin, screw, multi, loc, hin, mode]
    out = sum(w * x for w, x in zip(weights, parts))
    bs = float(p.get("bond_strength", 0.0))
    if bs > 1e-6:
        for i in range(out.shape[0]):
            out[i] = _restore_bonds(out[i], m, bs)
    return out


def run_interpolation(ca_start, ca_end, fracs, mask):
    return predict_interior(ca_start, ca_end, fracs, mask)
