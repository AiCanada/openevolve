"""Parameterized path interpolator — params evolved by run_param_evolve.py."""
import numpy as np

# EVOLVE-BLOCK-START
# Default params; overwritten by inject_params() for each trial.
PARAMS = {'nseg': 3, 'ease': 0.0, 'w_lin': 0.214060709927158, 'w_screw': 0.5053552881033624, 'w_multi': 0.0, 'bond_strength': 0.0, 'local_half': 11, 'use_local': 0.2586891601695809, 'hinge_smooth': 0, 'use_hinge': 0.15}


def inject_params(p: dict):
    PARAMS.update(p)


def _kabsch_R(P, Q):
    if P.shape[0] < 3:
        return np.eye(3)
    mu_p = P.mean(0)
    mu_q = Q.mean(0)
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


def _ease(s, amount):
    s = float(s)
    u = s * s * (3.0 - 2.0 * s)
    a = float(np.clip(amount, 0.0, 1.0))
    return (1.0 - a) * s + a * u


def _restore_bonds(x, mask, strength):
    if strength <= 1e-8:
        return x
    x = x.copy()
    idx = np.where(mask)[0]
    target, st = 3.8, float(strength)
    for _ in range(2):
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:
                continue
            d = x[b] - x[a]
            n = float(np.linalg.norm(d))
            if n < 1e-8:
                continue
            corr = (target - n) * st * 0.5
            u = d / n
            x[a] -= corr * u
            x[b] += corr * u
    return x


def _global_screw(ca0, ca1, m, fracs, ease):
    Ti, L = len(fracs), ca0.shape[0]
    out = np.empty((Ti, L, 3))
    w = m.astype(np.float64)
    sw = max(w.sum(), 1.0)
    mu0 = (w[:, None] * ca0).sum(0) / sw
    mu1 = (w[:, None] * ca1).sum(0) / sw
    if m.sum() < 3:
        for i, s in enumerate(fracs):
            out[i] = (1.0 - s) * ca0 + s * ca1
        return out
    R = _kabsch_R(ca0[m], ca1[m])
    omega = _so3_log(R)
    residual = ca1 - ((ca0 - mu0) @ R.T + mu1)
    for i, s in enumerate(fracs):
        s = float(s)
        u = _ease(s, ease)
        Rs = _so3_exp(omega * u)
        mus = (1.0 - s) * mu0 + s * mu1
        out[i] = (ca0 - mu0) @ Rs.T + mus + s * residual
    return out


def _multi_screw(ca0, ca1, m, fracs, nseg, ease):
    Ti, L = len(fracs), ca0.shape[0]
    base = _global_screw(ca0, ca1, m, fracs, ease)
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
        mu0 = ca0[sel].mean(0)
        mu1 = ca1[sel].mean(0)
        omega = _so3_log(R)
        residual = ca1[sel] - ((ca0[sel] - mu0) @ R.T + mu1)
        for i, s in enumerate(fracs):
            s = float(s)
            u = _ease(s, ease)
            Rs = _so3_exp(omega * u)
            mus = (1.0 - s) * mu0 + s * mu1
            out[i, sel] = (ca0[sel] - mu0) @ Rs.T + mus + s * residual
    return out


def _local_screw(ca0, ca1, m, fracs, half, ease):
    Ti, L = len(fracs), ca0.shape[0]
    out = np.empty((Ti, L, 3))
    half = max(2, int(half))
    for i in range(L):
        lo, hi = max(0, i - half), min(L, i + half + 1)
        sel = np.arange(lo, hi)
        sel = sel[m[sel]]
        if sel.size < 3:
            for t, s in enumerate(fracs):
                out[t, i] = (1.0 - s) * ca0[i] + s * ca1[i]
            continue
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0, mu1 = ca0[sel].mean(0), ca1[sel].mean(0)
        omega = _so3_log(R)
        res = ca1[i] - ((ca0[i] - mu0) @ R.T + mu1)
        for t, s in enumerate(fracs):
            s = float(s)
            u = _ease(s, ease)
            Rs = _so3_exp(omega * u)
            mus = (1.0 - s) * mu0 + s * mu1
            out[t, i] = (ca0[i] - mu0) @ Rs.T + mus + s * res
    return out


def _hinge_two(ca0, ca1, m, fracs, smooth, ease):
    Ti, L = len(fracs), ca0.shape[0]
    out = np.empty((Ti, L, 3))
    idx = np.where(m)[0]
    if idx.size < 10:
        return _global_screw(ca0, ca1, m, fracs, ease)
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
    sm = max(0, int(smooth))
    for i, s in enumerate(fracs):
        s = float(s)
        frame = (1.0 - s) * ca0 + s * ca1
        for sel, mu0, mu1, omega, residual in domains:
            u = _ease(s, ease)
            Rs = _so3_exp(omega * u)
            mus = (1.0 - s) * mu0 + s * mu1
            frame[sel] = (ca0[sel] - mu0) @ Rs.T + mus + s * residual
        for j in range(max(0, hinge - sm), min(L, hinge + sm + 1)):
            if m[j]:
                frame[j] = 0.5 * frame[j] + 0.5 * ((1.0 - s) * ca0[j] + s * ca1[j])
        out[i] = frame
    return out


def predict_interior(ca_start, ca_end, fracs, mask):
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    fracs = np.asarray(fracs, dtype=np.float64)
    p = PARAMS
    ease = float(p.get("ease", 0.0))
    lin = np.stack([(1.0 - s) * ca0 + s * ca1 for s in fracs], axis=0)
    screw = _global_screw(ca0, ca1, m, fracs, ease)
    multi = _multi_screw(ca0, ca1, m, fracs, p.get("nseg", 3), ease)

    w_lin = float(p.get("w_lin", 0.25))
    w_sc = float(p.get("w_screw", 0.35))
    w_mu = float(p.get("w_multi", 0.40))
    ssum = abs(w_lin) + abs(w_sc) + abs(w_mu) + 1e-12
    out = (w_lin * lin + w_sc * screw + w_mu * multi) / ssum

    ul = float(np.clip(p.get("use_local", 0.0), 0.0, 1.0))
    if ul > 1e-6:
        loc = _local_screw(ca0, ca1, m, fracs, p.get("local_half", 6), ease)
        out = (1.0 - ul) * out + ul * loc

    uh = float(np.clip(p.get("use_hinge", 0.0), 0.0, 1.0))
    if uh > 1e-6:
        hin = _hinge_two(ca0, ca1, m, fracs, p.get("hinge_smooth", 2), ease)
        out = (1.0 - uh) * out + uh * hin

    bs = float(p.get("bond_strength", 0.0))
    if bs > 1e-6:
        for i in range(out.shape[0]):
            out[i] = _restore_bonds(out[i], m, bs)
    return out


# EVOLVE-BLOCK-END


def run_interpolation(ca_start, ca_end, fracs, mask):
    return predict_interior(ca_start, ca_end, fracs, mask)
