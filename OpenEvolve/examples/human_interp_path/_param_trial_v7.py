
import numpy as np
PARAMS = {'ease': 0.15025106839013577, 'hermite': 0.11370771013186731, 'w_lin': 0.05461007778668368, 'w_screw': 0.3881616900985626, 'w_local': 0.4480661310744582, 'w_dom': 0.045079345361933754, 'w_mode': 0.0640827556773618, 'w_ms_small': 0.5944780734360332, 'w_ms_mid': 0.4055219265629669, 'w_ms_large': 0.0, 'use_chasles': 0.034091927148485454, 'timewarp': 0.0, 's_gate': 0.0007813990282099212, 'mode_mid_boost': 0.2956056785222733, 'bond_strength': 0.0, 'local_half': 28, 'half_s': 2, 'half_l': 37, 'n_dom': 2, 'min_dom': 8, 'n_modes': 1, 'algo_mode': 0}

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

def _local_flex(ca0, ca1, m, fracs, half_lo, half_hi, ease, herm):
    """Per-residue adaptive half-window from displacement magnitude."""
    L, Ti = ca0.shape[0], len(fracs)
    disp = np.linalg.norm(ca1 - ca0, axis=1)
    dmax = float(disp.max()) + 1e-8
    out = np.empty((Ti, L, 3))
    for i in range(L):
        if not m[i]:
            for ti, s in enumerate(fracs):
                t = _time(s, ease, herm)
                out[ti, i] = (1.0 - t) * ca0[i] + t * ca1[i]
            continue
        frac = float(disp[i] / dmax)
        half = int(round(half_lo + (half_hi - half_lo) * (1.0 - 0.65 * frac)))
        half = max(2, min(half, half_hi))
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

def _geodesic_lite(ca0, ca1, m, fracs, ease, herm, n_anchors=48):
    """Lightweight distance-geodesic: subsample anchors, lerp pairwise dists, embed via
    classical MDS on anchors, then RBF-map remaining residues (approx)."""
    L, Ti = ca0.shape[0], len(fracs)
    idx = np.where(m)[0]
    if idx.size < 8:
        out = np.empty((Ti, L, 3))
        for i, s in enumerate(fracs):
            t = _time(s, ease, herm)
            out[i] = (1.0 - t) * ca0 + t * ca1
        return out
    # pick anchors uniformly along chain
    na = min(n_anchors, idx.size)
    pick = np.linspace(0, idx.size - 1, na).astype(int)
    aidx = idx[pick]
    X0, X1 = ca0[aidx], ca1[aidx]
    # pairwise distances
    def pdist(X):
        d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        return d
    D0, D1 = pdist(X0), pdist(X1)
    out = np.empty((Ti, L, 3))
    # full path for anchors via lerp of endpoints (fallback blend with MDS if stable)
    for ti, s in enumerate(fracs):
        t = _time(s, ease, herm)
        Dt = (1.0 - t) * D0 + t * D1
        # classical MDS (3D) on anchors
        n = Dt.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (Dt ** 2) @ J
        try:
            evals, evecs = np.linalg.eigh(B)
            order = np.argsort(evals)[::-1]
            evals = np.maximum(evals[order[:3]], 0.0)
            coords = evecs[:, order[:3]] * np.sqrt(evals)[None, :]
            # Procrustes to linear mid for orientation
            mid = (1.0 - t) * X0 + t * X1
            R = _kabsch_R(coords, mid)
            mu_c, mu_m = coords.mean(0), mid.mean(0)
            anchors = (coords - mu_c) @ R.T + mu_m
        except np.linalg.LinAlgError:
            anchors = (1.0 - t) * X0 + t * X1
        # map all residues: local rigid from nearest anchors
        frame = (1.0 - t) * ca0 + t * ca1
        for i in range(L):
            if not m[i]:
                out[ti, i] = frame[i]
                continue
            # 3 nearest anchors
            d2 = np.sum((anchors - ca0[i]) ** 2, axis=1)  # rough
            # better: nearest in sequence index space among aidx
            di = np.abs(aidx.astype(np.float64) - float(i))
            nn = np.argsort(di)[: min(4, na)]
            sel = aidx[nn]
            if sel.size < 3:
                out[ti, i] = frame[i]
                continue
            Rloc = _kabsch_R(ca0[sel], anchors[nn] if len(nn) == sel.size else anchors[: sel.size])
            # use endpoint residual blend
            mu0s = ca0[sel].mean(0)
            mus = anchors[nn].mean(0) if len(nn) == sel.size else anchors[: sel.size].mean(0)
            out[ti, i] = (ca0[i] - mu0s) @ Rloc.T + mus
            # soft blend with linear to stabilize
            out[ti, i] = 0.65 * out[ti, i] + 0.35 * frame[i]
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

def _bond_restore(x, m, strength, n_pass=2):
    if strength < 1e-8:
        return x
    idx = np.where(m)[0]
    out = x.copy()
    for _ in range(n_pass):
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:
                continue
            d = out[b] - out[a]
            n = float(np.linalg.norm(d))
            if n < 1e-8:
                continue
            corr = (3.8 - n) * strength * 0.5
            u = d / n
            out[a] -= corr * u
            out[b] += corr * u
    return out

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
    s_gate = float(p.get("s_gate", 0.0))
    mode_mid = float(p.get("mode_mid_boost", 0.0))
    algo_mode = int(p.get("algo_mode", 0))

    # --- linear ---
    lin = np.empty((Ti, L, 3))
    for i, s in enumerate(fracs):
        t = _time(s, ease, herm)
        lin[i] = (1.0 - t) * ca0 + t * ca1

    # Mode 4: geodesic_lite early return (blend lightly with lin)
    if algo_mode == 4:
        geo = _geodesic_lite(ca0, ca1, m, fracs, ease, herm)
        alpha = float(np.clip(p.get("w_local", 0.5), 0.15, 0.9))
        out = (1.0 - alpha) * lin + alpha * geo
        bs = float(p.get("bond_strength", 0.05))
        if bs > 1e-6:
            for i in range(Ti):
                out[i] = _bond_restore(out[i], m, bs, n_pass=2)
        return out

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
            se = t * t * (3.0 - 2.0 * t)
            t_res = (1.0 - warp * alpha) * t + (warp * alpha) * se
            screw[i] = rigid + t_res[:, None] * residual_g

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
        uc = float(np.clip(use_chasles, 0.0, 1.0))
        screw = (1.0 - uc) * screw + uc * chasles

    # --- multi-scale / flex local ---
    if algo_mode == 1:
        # equal multiscale vote
        loc = (
            _local_scale(ca0, ca1, m, fracs, half_s, ease, herm)
            + _local_scale(ca0, ca1, m, fracs, half, ease, herm)
            + _local_scale(ca0, ca1, m, fracs, half_l, ease, herm)
        ) / 3.0
    elif algo_mode == 2:
        loc = _local_flex(ca0, ca1, m, fracs, half_s, half_l, ease, herm)
    else:
        w_ms = float(p.get("w_ms_small", 0.0))
        w_mm = float(p.get("w_ms_mid", 1.0))
        w_ml = float(p.get("w_ms_large", 0.0))
        ws_m = np.array([max(0.0, w_ms), max(0.0, w_mm), max(0.0, w_ml)], dtype=np.float64)
        if ws_m.sum() < 1e-12:
            ws_m[:] = [0.0, 1.0, 0.0]
        ws_m = ws_m / ws_m.sum()
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

    # --- correlation domains ---
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

    # --- blend weights (mode-specific biases) ---
    w_lin = float(p.get("w_lin", 0.1))
    w_screw = float(p.get("w_screw", 0.5))
    w_local = float(p.get("w_local", 0.2))
    w_dom = float(p.get("w_dom", 0.0))
    w_mode = float(p.get("w_mode", 0.0))
    if algo_mode == 1:
        # multiscale vote: favor local
        w_local = max(w_local, 0.45)
        w_screw = max(w_screw, 0.25)
    elif algo_mode == 2:
        w_local = max(w_local, 0.55)
        w_lin = min(w_lin, 0.08)
    elif algo_mode == 3:
        # bond_heavy: still blend but strong bond post
        w_local = max(w_local, 0.40)
    elif algo_mode == 5:
        # ensemble_local: 50/50 screw and local base, then mix with weights
        w_screw = 0.5 * w_screw + 0.25
        w_local = 0.5 * w_local + 0.25
    raw = np.array([w_lin, w_screw, w_local, w_dom, w_mode], dtype=np.float64)
    raw = np.maximum(raw, 0.0)
    if raw.sum() < 1e-12:
        raw[0] = 1.0
    raw = raw / raw.sum()

    out = np.empty((Ti, L, 3))
    for i, s in enumerate(fracs):
        s = float(s)
        mid = np.sin(np.pi * s)
        g = float(np.clip(s_gate, 0.0, 1.0)) * mid
        w = raw.copy()
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

    bs = float(p.get("bond_strength", 0.0))
    if algo_mode == 3:
        bs = max(bs, 0.08)
        n_pass = 4
    else:
        n_pass = 2
    if bs > 1e-6:
        for i in range(Ti):
            out[i] = _bond_restore(out[i], m, bs, n_pass=n_pass)
    return out

def run_interpolation(ca_start, ca_end, fracs, mask):
    return predict_interior(ca_start, ca_end, fracs, mask)
