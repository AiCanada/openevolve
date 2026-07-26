
import numpy as np
PARAMS = {'ease': 0, 'hermite': 0.0, 'w_lin': 0.10693815864018105, 'w_screw': 0.6613103922323285, 'w_local': 0.17530899683519152, 'w_dom': 0.0, 'w_mode': 0.05644245229130296, 'local_half': 20, 'half_s': 3, 'half_l': 22, 'w_ms_small': 0.13413454677601366, 'w_ms_mid': 0.8556340414937043, 'w_ms_large': 0.010231411729282176, 'n_dom': 1, 'min_dom': 9, 'use_chasles': 0.0, 'timewarp': 0.0, 's_gate': 0.14743748266883822, 'n_modes': 1, 'mode_mid_boost': 0.113, 'bond_strength': 0}

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
    sh = 3 * s * s - 2 * s * s * s
    se = s * s * (3.0 - 2.0 * s)
    t = (1.0 - h) * ((1.0 - e) * s + e * se) + h * sh
    return float(np.clip(t, 0.0, 1.0))

def _local_scale(ca0, ca1, m, fracs, half, ease, herm):
    L, Ti = ca0.shape[0], len(fracs)
    out = np.empty((Ti, L, 3))
    for i in range(L):
        lo, hi = max(0, i - half), min(L, i + half + 1)
        sel = np.arange(lo, hi)
        sel = sel[m[sel]]
        if sel.size < 3:
            for ti, s in enumerate(fracs):
                t = _time(s, ease, herm)
                out[ti, i] = (1.0 - t) * ca0[i] + t * ca1[i]
            continue
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0, mu1 = ca0[sel].mean(0), ca1[sel].mean(0)
        om = _so3_log(R)
        res = ca1[i] - ((ca0[i] - mu0) @ R.T + mu1)
        for ti, s in enumerate(fracs):
            t = _time(s, ease, herm)
            Rs = _so3_exp(om * t)
            mus = (1.0 - t) * mu0 + t * mu1
            out[ti, i] = (ca0[i] - mu0) @ Rs.T + mus + t * res
    return out

def _corr_domains(disp, m, n_dom, min_size):
    idx = np.where(m)[0]
    if idx.size < max(3, n_dom * min_size):
        return [idx] if idx.size else []
    norms = np.linalg.norm(disp, axis=1) + 1e-12
    u = disp / norms[:, None]
    breaks = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            breaks.append((a + 0.5, 2.0))
            continue
        c = float(np.dot(u[a], u[b]))
        mag = 0.5 * (norms[a] + norms[b])
        breaks.append((a + 0.5, (1.0 - c) * mag))
    breaks_sorted = sorted(breaks, key=lambda x: -x[1])
    chosen = []
    for pos, sc in breaks_sorted:
        if sc < 0.12:
            break
        if all(abs(pos - c) > min_size for c in chosen):
            chosen.append(pos)
        if len(chosen) >= max(0, n_dom - 1):
            break
    cuts = sorted(int(np.ceil(c)) for c in chosen)
    start, end = int(idx[0]), int(idx[-1]) + 1
    bounds = [start] + cuts + [end]
    domains = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        sel = np.arange(a, b)
        sel = sel[m[sel]]
        if sel.size >= 3:
            domains.append(sel)
    return domains if domains else [idx]

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
    half_s = max(2, int(p.get("half_s", 5)))
    half_l = max(half + 2, int(p.get("half_l", 22)))
    n_dom = max(1, int(p.get("n_dom", 3)))
    min_dom = max(4, int(p.get("min_dom", 8)))
    use_chasles = float(p.get("use_chasles", 1.0))
    warp = float(p.get("timewarp", 0.0))
    s_gate = float(p.get("s_gate", 0.0))  # how much mid-boost local/screw
    mode_mid = float(p.get("mode_mid_boost", 0.0))

    # --- linear ---
    lin = np.empty((Ti, L, 3))
    for i, s in enumerate(fracs):
        t = _time(s, ease, herm)
        lin[i] = (1.0 - t) * ca0 + t * ca1

    # --- standard global screw + residual ---
    screw = lin.copy()
    chasles = lin.copy()
    residual_g = np.zeros((L, 3))
    alpha = np.zeros(L)
    if m.sum() >= 3:
        R = _kabsch_R(ca0[m], ca1[m])
        mu0 = ca0[m].mean(0)
        mu1 = ca1[m].mean(0)
        omega = _so3_log(R)
        residual_g = ca1 - ((ca0 - mu0) @ R.T + mu1)
        disp = np.linalg.norm(ca1 - ca0, axis=1)
        dmax = float(disp.max()) + 1e-8
        alpha = np.clip(disp / dmax, 0.0, 1.0) * m.astype(np.float64)
        # smooth alpha
        a2 = alpha.copy()
        for _ in range(2):
            a2[1:-1] = 0.25 * alpha[:-2] + 0.5 * alpha[1:-1] + 0.25 * alpha[2:]
            alpha = a2.copy()
            alpha = alpha * m.astype(np.float64)

        for i, s in enumerate(fracs):
            t = _time(s, ease, herm)
            Rs = _so3_exp(omega * t)
            mus = (1.0 - t) * mu0 + t * mu1
            rigid = (ca0 - mu0) @ Rs.T + mus
            # timewarp residual for high-disp residues
            se = t * t * (3.0 - 2.0 * t)
            t_res = (1.0 - warp * alpha) * t + (warp * alpha) * se
            screw[i] = rigid + t_res[:, None] * residual_g

        # Chasles true screw
        tvec = mu1 - R @ mu0
        theta = float(np.linalg.norm(omega))
        if theta < 1e-8:
            chasles = screw.copy()
        else:
            k = omega / theta
            t_par = float(np.dot(tvec, k)) * k
            t_perp = tvec - t_par
            tan_half = np.tan(theta * 0.5)
            if abs(tan_half) < 1e-10:
                pax = mu0
            else:
                pax = 0.5 * t_perp + np.cross(k, tvec) / (2.0 * tan_half)
            rigid_end = (ca0 - pax) @ R.T + pax + t_par
            res_c = ca1 - rigid_end
            for i, s in enumerate(fracs):
                t = _time(s, ease, herm)
                Rs = _so3_exp(omega * t)
                rigid = (ca0 - pax) @ Rs.T + pax + t * t_par
                se = t * t * (3.0 - 2.0 * t)
                t_res = (1.0 - warp * alpha) * t + (warp * alpha) * se
                chasles[i] = rigid + t_res[:, None] * res_c
        # blend standard vs chasles
        uc = float(np.clip(use_chasles, 0.0, 1.0))
        screw = (1.0 - uc) * screw + uc * chasles

    # --- multi-scale local (skip unused scales for speed) ---
    w_ms = float(p.get("w_ms_small", 0.0))
    w_mm = float(p.get("w_ms_mid", 1.0))
    w_ml = float(p.get("w_ms_large", 0.0))
    ws_m = np.array([max(0.0, w_ms), max(0.0, w_mm), max(0.0, w_ml)], dtype=np.float64)
    if ws_m.sum() < 1e-12:
        ws_m[:] = [0.0, 1.0, 0.0]
    ws_m = ws_m / ws_m.sum()
    # default: single mid scale (matches v3 local)
    if ws_m[0] < 0.02 and ws_m[2] < 0.02:
        loc = _local_scale(ca0, ca1, m, fracs, half, ease, herm)
    else:
        loc = np.zeros((Ti, L, 3))
        if ws_m[0] >= 0.02:
            loc = loc + ws_m[0] * _local_scale(ca0, ca1, m, fracs, half_s, ease, herm)
        if ws_m[1] >= 0.02:
            loc = loc + ws_m[1] * _local_scale(ca0, ca1, m, fracs, half, ease, herm)
        if ws_m[2] >= 0.02:
            loc = loc + ws_m[2] * _local_scale(ca0, ca1, m, fracs, half_l, ease, herm)

    # --- correlation domains (skip if unused) ---
    w_dom_pre = float(p.get("w_dom", 0.0))
    dom = screw
    if w_dom_pre > 0.02 and m.sum() >= 12 and n_dom >= 2:
        dom = screw.copy()
        domains = _corr_domains(ca1 - ca0, m, n_dom, min_dom)
        for sel in domains:
            R = _kabsch_R(ca0[sel], ca1[sel])
            mu0s, mu1s = ca0[sel].mean(0), ca1[sel].mean(0)
            om = _so3_log(R)
            res = ca1[sel] - ((ca0[sel] - mu0s) @ R.T + mu1s)
            for i, s in enumerate(fracs):
                t = _time(s, ease, herm)
                Rs = _so3_exp(om * t)
                mus = (1.0 - t) * mu0s + t * mu1s
                dom[i, sel] = (ca0[sel] - mu0s) @ Rs.T + mus + t * res

    # --- light PCA residual mode path ---
    mode_path = lin
    nmodes = max(0, int(p.get("n_modes", 0)))
    w_mode_pre = float(p.get("w_mode", 0.0))
    if nmodes > 0 and w_mode_pre > 0.02 and m.sum() >= 6:
        mode_path = lin.copy()
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
                t = _time(s, ease, herm)
                mid = mode_mid * np.sin(np.pi * s)
                delta = np.zeros((L, 3))
                delta[idxm] = (t + mid) * recon + t * miss
                mode_path[i] = ca0 + delta
        except np.linalg.LinAlgError:
            pass

    # --- base static weights ---
    w_lin = float(p.get("w_lin", 0.1))
    w_screw = float(p.get("w_screw", 0.5))
    w_local = float(p.get("w_local", 0.2))
    w_dom = float(p.get("w_dom", 0.0))
    w_mode = float(p.get("w_mode", 0.0))
    raw = np.array([w_lin, w_screw, w_local, w_dom, w_mode], dtype=np.float64)
    raw = np.maximum(raw, 0.0)
    if raw.sum() < 1e-12:
        raw[0] = 1.0
    raw = raw / raw.sum()

    # s-dependent reweight: at mid, push weight from lin -> screw/local
    out = np.empty((Ti, L, 3))
    for i, s in enumerate(fracs):
        s = float(s)
        mid = np.sin(np.pi * s)
        g = float(np.clip(s_gate, 0.0, 1.0)) * mid
        w = raw.copy()
        # transfer up to g*0.25 from lin to screw and local
        take = min(w[0], g * 0.25)
        w[0] -= take
        w[1] += take * 0.65
        w[2] += take * 0.35
        w = w / (w.sum() + 1e-12)
        out[i] = (
            w[0] * lin[i]
            + w[1] * screw[i]
            + w[2] * loc[i]
            + w[3] * dom[i]
            + w[4] * mode_path[i]
        )

    # optional gentle bond restore
    bs = float(p.get("bond_strength", 0.0))
    if bs > 1e-6:
        idx = np.where(m)[0]
        for i in range(Ti):
            x = out[i].copy()
            for _ in range(2):
                for a, b in zip(idx[:-1], idx[1:]):
                    if b != a + 1:
                        continue
                    d = x[b] - x[a]
                    n = float(np.linalg.norm(d))
                    if n < 1e-8:
                        continue
                    corr = (3.8 - n) * bs * 0.5
                    u = d / n
                    x[a] -= corr * u
                    x[b] += corr * u
            out[i] = x
    return out

def run_interpolation(ca_start, ca_end, fracs, mask):
    return predict_interior(ca_start, ca_end, fracs, mask)
