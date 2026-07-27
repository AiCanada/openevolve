"""Per-residue time warp: high-displacement residues progress with smoother mid schedule."""
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


def predict_interior(ca_start, ca_end, fracs, mask):
    ca0 = np.asarray(ca_start, float)
    ca1 = np.asarray(ca_end, float)
    m = np.asarray(mask, bool)
    fracs = np.asarray(fracs, float)
    L, Ti = ca0.shape[0], len(fracs)

    disp = np.linalg.norm(ca1 - ca0, axis=1)
    dmax = float(disp.max()) + 1e-8
    # alpha in [0,1]: how much smoothstep timewarp for this residue
    alpha = np.clip(disp / dmax, 0.0, 1.0)
    # lightly smooth along chain
    a2 = alpha.copy()
    for _ in range(2):
        a2[1:-1] = 0.25 * alpha[:-2] + 0.5 * alpha[1:-1] + 0.25 * alpha[2:]
        alpha = a2.copy()
    alpha = alpha * m.astype(float)

    if m.sum() < 3:
        out = np.empty((Ti, L, 3))
        for i, s in enumerate(fracs):
            out[i] = (1 - s) * ca0 + s * ca1
        return out

    R = _kabsch_R(ca0[m], ca1[m])
    mu0, mu1 = ca0[m].mean(0), ca1[m].mean(0)
    om = _so3_log(R)
    # endpoint residual after full screw
    residual = ca1 - ((ca0 - mu0) @ R.T + mu1)

    out = np.empty((Ti, L, 3))
    half = 13
    for ti, s in enumerate(fracs):
        s = float(s)
        # global rigid with shared time
        Rs = _so3_exp(om * s)
        mus = (1 - s) * mu0 + s * mu1
        rigid = (ca0 - mu0) @ Rs.T + mus

        # per-residue residual progress: mix linear s with smoothstep for high-disp
        se = s * s * (3.0 - 2.0 * s)
        t_res = (1.0 - 0.35 * alpha) * s + (0.35 * alpha) * se
        frame = rigid + t_res[:, None] * residual

        # light local correction blend for mid-path only
        mid = np.sin(np.pi * s)
        if mid > 0.05:
            loc = frame.copy()
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
                oml = _so3_log(Rl)
                resi = ca1[i] - ((ca0[i] - m0) @ Rl.T + m1)
                Rsl = _so3_exp(oml * s)
                musl = (1 - s) * m0 + s * m1
                loc[i] = (ca0[i] - m0) @ Rsl.T + musl + s * resi
            w_loc = 0.18 * mid
            frame = (1 - w_loc) * frame + w_loc * loc
        out[ti] = frame
    return out
# EVOLVE-BLOCK-END


def run_interpolation(ca_start, ca_end, fracs, mask):
    return predict_interior(ca_start, ca_end, fracs, mask)
