
import numpy as np
PARAMS = {'w_lin': 0.10388349514563107, 'w_screw': 0.641747572815534, 'w_local': 0.2, 'w_mode': 0.05436893203883496, 'local_half': 20, 'half_s': 3, 'half_lo': 8, 'half_hi': 22, 'w_ms_small': 0.0, 's_gate': 0.1, 'agree_str': 0.0, 'use_flex': 0.0, 'mag_mid': 0.0, 'n_modes': 1, 'mode_mid_boost': 0.113}

def inject_params(p):
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
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (theta / (2.0 * np.sin(theta) + 1e-12))

def _so3_exp(w):
    theta = float(np.linalg.norm(w))
    if theta < 1e-10:
        return np.eye(3)
    k = w / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

def _local_scale(ca0, ca1, m, fracs, half):
    L, Ti = ca0.shape[0], len(fracs)
    out = np.empty((Ti, L, 3))
    for i in range(L):
        lo, hi = max(0, i - half), min(L, i + half + 1)
        sel = np.arange(lo, hi)
        sel = sel[m[sel]]
        if sel.size < 3:
            for ti, s in enumerate(fracs):
                out[ti, i] = (1.0 - s) * ca0[i] + s * ca1[i]
            continue
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0, mu1 = ca0[sel].mean(0), ca1[sel].mean(0)
        om = _so3_log(R)
        res = ca1[i] - ((ca0[i] - mu0) @ R.T + mu1)
        for ti, s in enumerate(fracs):
            s = float(s)
            Rs = _so3_exp(om * s)
            mus = (1.0 - s) * mu0 + s * mu1
            out[ti, i] = (ca0[i] - mu0) @ Rs.T + mus + s * res
    return out

def _local_flex(ca0, ca1, m, fracs, half_lo, half_hi, res_g):
    """Per-residue half from residual magnitude."""
    L, Ti = ca0.shape[0], len(fracs)
    rmag = np.linalg.norm(res_g, axis=1)
    r = rmag / (rmag.max() + 1e-8)
    half_i = (half_hi - (half_hi - half_lo) * r).astype(int)
    half_i = np.clip(half_i, half_lo, half_hi)
    out = np.empty((Ti, L, 3))
    for i in range(L):
        half = int(half_i[i])
        lo, hi = max(0, i - half), min(L, i + half + 1)
        sel = np.arange(lo, hi)
        sel = sel[m[sel]]
        if sel.size < 3:
            for ti, s in enumerate(fracs):
                out[ti, i] = (1.0 - s) * ca0[i] + s * ca1[i]
            continue
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0, mu1 = ca0[sel].mean(0), ca1[sel].mean(0)
        om = _so3_log(R)
        res = ca1[i] - ((ca0[i] - mu0) @ R.T + mu1)
        for ti, s in enumerate(fracs):
            s = float(s)
            Rs = _so3_exp(om * s)
            mus = (1.0 - s) * mu0 + s * mu1
            out[ti, i] = (ca0[i] - mu0) @ Rs.T + mus + s * res
    return out

def predict_interior(ca_start, ca_end, fracs, mask):
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    fracs = np.asarray(fracs, dtype=np.float64)
    p = PARAMS
    L = ca0.shape[0]
    Ti = len(fracs)
    half = max(2, int(p.get("local_half", 16)))
    half_s = max(2, int(p.get("half_s", 4)))
    half_lo = max(4, int(p.get("half_lo", 8)))
    half_hi = max(half_lo + 2, int(p.get("half_hi", 20)))
    s_gate = float(p.get("s_gate", 0.0))
    agree_str = float(p.get("agree_str", 0.0))
    use_flex = float(p.get("use_flex", 0.0))
    mag_mid = float(p.get("mag_mid", 0.0))
    mode_mid = float(p.get("mode_mid_boost", 0.0))
    w_ms_s = float(p.get("w_ms_small", 0.0))

    lin = np.empty((Ti, L, 3))
    for i, s in enumerate(fracs):
        s = float(s)
        lin[i] = (1.0 - s) * ca0 + s * ca1

    res_g = np.zeros((L, 3))
    screw = lin.copy()
    if m.sum() >= 3:
        R = _kabsch_R(ca0[m], ca1[m])
        mu0 = ca0[m].mean(0)
        mu1 = ca1[m].mean(0)
        omega = _so3_log(R)
        res_g = ca1 - ((ca0 - mu0) @ R.T + mu1)
        mag = np.linalg.norm(res_g, axis=1, keepdims=True)
        u = np.divide(res_g, mag + 1e-12)
        for i, s in enumerate(fracs):
            s = float(s)
            Rs = _so3_exp(omega * s)
            mus = (1.0 - s) * mu0 + s * mu1
            rigid = (ca0 - mu0) @ Rs.T + mus
            mag_s = (s + mag_mid * np.sin(np.pi * s)) * mag
            screw[i] = rigid + mag_s * u

    # local: fixed half vs flex-adaptive
    if use_flex > 0.5 and m.sum() >= 3:
        loc = _local_flex(ca0, ca1, m, fracs, half_lo, half_hi, res_g)
    else:
        loc_m = _local_scale(ca0, ca1, m, fracs, half)
        if w_ms_s > 0.02:
            loc_s = _local_scale(ca0, ca1, m, fracs, half_s)
            loc = (1.0 - w_ms_s) * loc_m + w_ms_s * loc_s
        else:
            loc = loc_m

    # agree-gate: boost local where local rigid endpoint differs from global rigid
    w_loc_base = float(p.get("w_local", 0.15))
    w_loc_i = np.full(L, w_loc_base, dtype=np.float64)
    if agree_str > 1e-6 and m.sum() >= 3:
        R = _kabsch_R(ca0[m], ca1[m])
        mu0 = ca0[m].mean(0)
        mu1 = ca1[m].mean(0)
        agree = np.zeros(L)
        for i in range(L):
            if not m[i]:
                continue
            lo, hi = max(0, i - half), min(L, i + half + 1)
            sel = np.arange(lo, hi)
            sel = sel[m[sel]]
            if sel.size < 3:
                continue
            Rl = _kabsch_R(ca0[sel], ca1[sel])
            m0, m1 = ca0[sel].mean(0), ca1[sel].mean(0)
            glob_rig = (ca0[i] - mu0) @ R.T + mu1
            loc_rig = (ca0[i] - m0) @ Rl.T + m1
            agree[i] = float(np.linalg.norm(loc_rig - glob_rig))
        amax = float(agree.max()) + 1e-8
        an = agree / amax
        w_loc_i = w_loc_base + agree_str * an
        for _ in range(2):
            w2 = w_loc_i.copy()
            w2[1:-1] = 0.25 * w_loc_i[:-2] + 0.5 * w_loc_i[1:-1] + 0.25 * w_loc_i[2:]
            w_loc_i = w2
        w_loc_i = w_loc_i * m.astype(np.float64)

    # mode path
    mode_path = lin.copy()
    nmodes = max(0, int(p.get("n_modes", 0)))
    w_mode = float(p.get("w_mode", 0.0))
    if nmodes > 0 and w_mode > 0.02 and m.sum() >= 6:
        idxm = np.where(m)[0]
        D = ca1[idxm] - ca0[idxm]
        try:
            U, S, Vt = np.linalg.svd(D, full_matrices=False)
            k = min(nmodes, U.shape[1], 3)
            recon = np.zeros_like(D)
            for j in range(k):
                recon += np.outer(U[:, j] * S[j], Vt[j])
            miss = D - recon
            for i, s in enumerate(fracs):
                s = float(s)
                mid = mode_mid * np.sin(np.pi * s)
                delta = np.zeros((L, 3))
                delta[idxm] = (s + mid) * recon + s * miss
                mode_path[i] = ca0 + delta
        except np.linalg.LinAlgError:
            pass

    w_lin = float(p.get("w_lin", 0.1))
    w_screw = float(p.get("w_screw", 0.6))
    # if agree-gate active, per-residue weights; else static
    raw_static = np.array([w_lin, w_screw, w_loc_base, w_mode], dtype=np.float64)
    raw_static = np.maximum(raw_static, 0.0)
    if raw_static.sum() < 1e-12:
        raw_static[0] = 1.0
    raw_static = raw_static / raw_static.sum()

    out = np.empty((Ti, L, 3))
    for i, s in enumerate(fracs):
        s = float(s)
        mid = np.sin(np.pi * s)
        if agree_str > 1e-6:
            # per-residue: w_loc_i, redistribute remaining to lin/screw/mode
            wl = np.clip(w_loc_i, 0.0, 0.55)
            rem = np.clip(1.0 - wl, 0.05, 1.0)
            # base shares among non-local
            base = np.array([w_lin, w_screw, max(w_mode, 0.0)], dtype=np.float64)
            base = np.maximum(base, 0.0)
            if base.sum() < 1e-12:
                base = np.array([0.2, 0.8, 0.0])
            base = base / base.sum()
            # s-gate on non-local: push lin -> screw
            b = base.copy()
            take = min(b[0], s_gate * mid * 0.25)
            b[0] -= take
            b[1] += take
            b = b / (b.sum() + 1e-12)
            wn = rem * b[0]
            ws = rem * b[1]
            wm = rem * b[2]
            out[i] = (
                wn[:, None] * lin[i]
                + ws[:, None] * screw[i]
                + wl[:, None] * loc[i]
                + wm[:, None] * mode_path[i]
            )
        else:
            w = raw_static.copy()
            take = min(w[0], s_gate * mid * 0.25)
            w[0] -= take
            w[1] += take * 0.65
            w[2] += take * 0.35
            w = w / (w.sum() + 1e-12)
            out[i] = w[0] * lin[i] + w[1] * screw[i] + w[2] * loc[i] + w[3] * mode_path[i]
    return out

def run_interpolation(ca_start, ca_end, fracs, mask):
    return predict_interior(ca_start, ca_end, fracs, mask)
