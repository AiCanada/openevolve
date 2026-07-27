
import numpy as np
PARAMS = {'nseg': 1, 'n_hinge': 3, 'n_modes': 1, 'ease': 0.0, 'hermite': 0.0, 'w_lin': 0.107235094938249, 'w_screw': 0.6449458580638288, 'w_multi': 0.0419144159119635, 'w_local': 0.14770667492125836, 'w_hinge': 0.0, 'w_mode': 0.058197956163683236, 'bond_strength': 0, 'local_half': 13, 'soft_local': 0.0, 'res_mid_boost': 0, 'mode_mid_boost': 0.11294738655336786}

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
    nhinge = max(1, int(p.get("n_hinge", 1)))
    nmodes = max(0, int(p.get("n_modes", 1)))
    soft = float(p.get("soft_local", 0.0))  # soft weighting by displacement

    # --- components ---
    lin = np.empty((Ti, L, 3))
    for i, s in enumerate(fracs):
        t = _time(s, ease, herm)
        lin[i] = (1.0 - t) * ca0 + t * ca1

    # global screw + residual
    screw = np.empty((Ti, L, 3))
    if m.sum() >= 3:
        w = m.astype(np.float64)
        sw = w.sum()
        mu0 = (w[:, None] * ca0).sum(0) / sw
        mu1 = (w[:, None] * ca1).sum(0) / sw
        R = _kabsch_R(ca0[m], ca1[m])
        omega = _so3_log(R)
        residual = ca1 - ((ca0 - mu0) @ R.T + mu1)
        rmag = np.linalg.norm(residual, axis=1)
        rmag = rmag / (rmag.max() + 1e-8)
        beta = float(p.get("res_mid_boost", 0.0))
        for i, s in enumerate(fracs):
            t = _time(s, ease, herm)
            Rs = _so3_exp(omega * t)
            mus = (1.0 - t) * mu0 + t * mu1
            rigid = (ca0 - mu0) @ Rs.T + mus
            prog = t + beta * np.sin(np.pi * s) * rmag
            screw[i] = rigid + prog[:, None] * residual
    else:
        screw = lin.copy()

    # multi-segment
    multi = screw.copy()
    idx = np.where(m)[0]
    if idx.size >= nseg * 3:
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

    # local window screw
    loc = np.empty((Ti, L, 3))
    disp = np.linalg.norm(ca1 - ca0, axis=1)
    for i in range(L):
        lo, hi = max(0, i - half), min(L, i + half + 1)
        sel = np.arange(lo, hi)
        sel = sel[m[sel]]
        if sel.size < 3:
            loc[:, i] = lin[:, i]
            continue
        if soft > 1e-6:
            # soft weights by inverse distance in sequence * displacement similarity
            center = i
            ww = np.exp(-0.5 * ((sel - center) / max(half / 2, 1.0)) ** 2)
            ww = ww * (0.2 + disp[sel])
            ww = ww / (ww.sum() + 1e-12)
            # weighted Kabsch
            mu0s = (ww[:, None] * ca0[sel]).sum(0)
            mu1s = (ww[:, None] * ca1[sel]).sum(0)
            X = (ca0[sel] - mu0s) * ww[:, None]
            Y = ca1[sel] - mu1s
            H = X.T @ Y
            try:
                U, _S, Vt = np.linalg.svd(H)
                d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
                R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
            except np.linalg.LinAlgError:
                R = np.eye(3)
        else:
            R = _kabsch_R(ca0[sel], ca1[sel])
            mu0s, mu1s = ca0[sel].mean(0), ca1[sel].mean(0)
        om = _so3_log(R)
        res = ca1[i] - ((ca0[i] - mu0s) @ R.T + mu1s)
        for ti, s in enumerate(fracs):
            t = _time(s, ease, herm)
            Rs = _so3_exp(om * t)
            mus = (1.0 - t) * mu0s + t * mu1s
            loc[ti, i] = (ca0[i] - mu0s) @ Rs.T + mus + t * res

    # multi-hinge domains by displacement gradient peaks
    hinge_path = screw.copy()
    if idx.size >= 12 and nhinge >= 1:
        d = disp.copy()
        grad = np.zeros(L)
        for a, b in zip(idx[:-1], idx[1:]):
            if b == a + 1:
                grad[b] = abs(d[b] - d[a])
        # pick top nhinge hinge residues along chain
        cand = idx[3:-3]
        order = cand[np.argsort(-grad[cand])]
        hinges = []
        for h in order:
            if all(abs(int(h) - int(hh)) > max(5, L // (nhinge * 4)) for hh in hinges):
                hinges.append(int(h))
            if len(hinges) >= nhinge:
                break
        hinges = sorted(hinges)
        bounds = [int(idx[0])] + hinges + [int(idx[-1]) + 1]
        domains = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            sel = np.arange(a, b)
            sel = sel[m[sel]]
            if sel.size < 3:
                continue
            R = _kabsch_R(ca0[sel], ca1[sel])
            mu0s, mu1s = ca0[sel].mean(0), ca1[sel].mean(0)
            om = _so3_log(R)
            res = ca1[sel] - ((ca0[sel] - mu0s) @ R.T + mu1s)
            domains.append((sel, mu0s, mu1s, om, res))
        for i, s in enumerate(fracs):
            t = _time(s, ease, herm)
            frame = lin[i].copy()
            for sel, mu0s, mu1s, om, res in domains:
                Rs = _so3_exp(om * t)
                mus = (1.0 - t) * mu0s + t * mu1s
                frame[sel] = (ca0[sel] - mu0s) @ Rs.T + mus + t * res
            hinge_path[i] = frame

    # residual PCA modes (top nmodes of endpoint displacement on masked residues)
    mode_path = lin.copy()
    if nmodes > 0 and m.sum() >= 6:
        idxm = np.where(m)[0]
        D = (ca1[idxm] - ca0[idxm])  # (n,3)
        # SVD of displacement as n x 3
        try:
            U, S, Vt = np.linalg.svd(D, full_matrices=False)
        except np.linalg.LinAlgError:
            U, S, Vt = None, None, None
        if U is not None:
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

    # blend
    ws = np.array([
        float(p.get("w_lin", 0.1)),
        float(p.get("w_screw", 0.5)),
        float(p.get("w_multi", 0.0)),
        float(p.get("w_local", 0.2)),
        float(p.get("w_hinge", 0.0)),
        float(p.get("w_mode", 0.0)),
    ], dtype=np.float64)
    ws = np.maximum(ws, 0.0)
    if ws.sum() < 1e-12:
        ws[0] = 1.0
    ws = ws / ws.sum()
    parts = [lin, screw, multi, loc, hinge_path, mode_path]
    out = sum(w * x for w, x in zip(ws, parts))

    # gentle bond restore
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
