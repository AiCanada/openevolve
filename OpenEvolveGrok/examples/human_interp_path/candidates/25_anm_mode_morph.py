"""ANM-like morph: project endpoint displacement onto low-frequency structure modes."""
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


def _anm_modes(coords, mask, cutoff=12.0, n_modes=6):
    """Crude anisotropic network: 3N Hessian from distance springs on start structure."""
    idx = np.where(mask)[0]
    n = idx.size
    if n < 6:
        return None, None, None
    X = coords[idx]
    # build sparse-ish dense Hessian (3n x 3n) — cap n for speed
    if n > 180:
        # subsample every k residues
        step = int(np.ceil(n / 180))
        idx = idx[::step]
        X = coords[idx]
        n = idx.size
    H = np.zeros((3 * n, 3 * n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = X[j] - X[i]
            r2 = float(np.dot(d, d))
            if r2 < 1e-8 or r2 > cutoff * cutoff:
                continue
            # unit spring stiffness 1/r^2
            k = 1.0 / r2
            # outer product block
            block = k * np.outer(d, d) / r2
            ii, jj = 3 * i, 3 * j
            H[ii:ii + 3, ii:ii + 3] += block
            H[jj:jj + 3, jj:jj + 3] += block
            H[ii:ii + 3, jj:jj + 3] -= block
            H[jj:jj + 3, ii:ii + 3] -= block
    try:
        # eigh; skip 6 near-zero rigid modes
        evals, evecs = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        return None, None, None
    # take lowest positive modes
    order = np.argsort(evals)
    modes = []
    for o in order:
        if evals[o] < 1e-8:
            continue
        modes.append(evecs[:, o])
        if len(modes) >= n_modes:
            break
    if not modes:
        return None, None, None
    M = np.stack(modes, axis=1)  # (3n, k)
    return idx, M, evals


def predict_interior(ca_start, ca_end, fracs, mask):
    ca0 = np.asarray(ca_start, float)
    ca1 = np.asarray(ca_end, float)
    m = np.asarray(mask, bool)
    fracs = np.asarray(fracs, float)
    L, Ti = ca0.shape[0], len(fracs)

    lin = np.stack([(1 - s) * ca0 + s * ca1 for s in fracs], 0)
    if m.sum() < 6:
        return lin

    # global screw baseline
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

    # ANM mode morph of residual
    idx, M, _ = _anm_modes(ca0, m, cutoff=12.0, n_modes=8)
    mode_path = screw.copy()
    if idx is not None and M is not None:
        disp = res_g[idx].reshape(-1)  # residual in global frame coords
        # project residual onto modes
        coeffs = M.T @ disp  # (k,)
        recon = (M @ coeffs).reshape(-1, 3)
        miss = res_g[idx] - recon
        for i, s in enumerate(fracs):
            s = float(s)
            # soft mid boost on low modes only
            mid = 0.08 * np.sin(np.pi * s)
            delta = np.zeros((L, 3))
            delta[idx] = (s + mid) * recon + s * miss
            # path = rigid screw without residual + mode residual
            Rs = _so3_exp(omg * s)
            mus = (1 - s) * mu0 + s * mu1
            rigid = (ca0 - mu0) @ Rs.T + mus
            mode_path[i] = rigid + delta

    # light local
    half = 14
    loc = np.empty((Ti, L, 3))
    for i in range(L):
        lo, hi = max(0, i - half), min(L, i + half + 1)
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

    return 0.12 * lin + 0.50 * screw + 0.20 * mode_path + 0.18 * loc
# EVOLVE-BLOCK-END


def run_interpolation(ca_start, ca_end, fracs, mask):
    return predict_interior(ca_start, ca_end, fracs, mask)
