"""Generate algorithm candidate .py files for local Grok evolution (no Claude, no xAI key)."""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

HEADER = '''"""Transition-path interpolation for protein backbones (Project H.U.M.A.N.)."""
import numpy as np

# EVOLVE-BLOCK-START

'''

FOOTER = '''
# EVOLVE-BLOCK-END


def run_interpolation(ca_start, ca_end, fracs, mask):
    return predict_interior(ca_start, ca_end, fracs, mask)
'''

SHARED = '''
def _kabsch_R(P, Q, w=None):
    if w is None:
        w = np.ones(P.shape[0], dtype=np.float64)
    w = w.astype(np.float64)
    sw = w.sum()
    if sw < 1e-12 or P.shape[0] < 3:
        return np.eye(3)
    mu_p = (w[:, None] * P).sum(0) / sw
    mu_q = (w[:, None] * Q).sum(0) / sw
    X = (P - mu_p) * w[:, None]
    Y = Q - mu_q
    H = X.T @ Y
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
        return np.zeros(3, dtype=np.float64)
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
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    return w * (theta / (2.0 * np.sin(theta) + 1e-12))


def _so3_exp(w):
    theta = float(np.linalg.norm(w))
    if theta < 1e-10:
        return np.eye(3)
    k = w / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _restore_ca_bonds(coords, mask, target=3.8, strength=0.35):
    """Pull consecutive Cα spacing toward ~3.8 A (gentle, keeps overall shape)."""
    x = coords.copy()
    idx = np.where(mask)[0]
    if idx.size < 2:
        return x
    for _ in range(2):
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:
                continue  # only sequential chain neighbors
            d = x[b] - x[a]
            n = float(np.linalg.norm(d))
            if n < 1e-8:
                continue
            corr = (target - n) * strength * 0.5
            u = d / n
            x[a] = x[a] - corr * u
            x[b] = x[b] + corr * u
    return x
'''

CANDIDATES = {
    "00_linear": '''
def predict_interior(ca_start, ca_end, fracs, mask):
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    out = np.empty((len(fracs), ca0.shape[0], 3), dtype=np.float64)
    for i, s in enumerate(fracs):
        out[i] = (1.0 - s) * ca0 + s * ca1
    return out
''',
    "01_screw_residual": SHARED + '''
def predict_interior(ca_start, ca_end, fracs, mask):
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    L = ca0.shape[0]
    Ti = len(fracs)
    out = np.empty((Ti, L, 3), dtype=np.float64)
    if m.sum() < 3:
        for i, s in enumerate(fracs):
            out[i] = (1.0 - s) * ca0 + s * ca1
        return out
    w = m.astype(np.float64)
    sw = w.sum()
    mu0 = (w[:, None] * ca0).sum(0) / sw
    mu1 = (w[:, None] * ca1).sum(0) / sw
    R = _kabsch_R(ca0[m], ca1[m])
    omega = _so3_log(R)
    rigid_end = (ca0 - mu0) @ R.T + mu1
    residual = ca1 - rigid_end
    for i, s in enumerate(fracs):
        s = float(s)
        Rs = _so3_exp(omega * s)
        mus = (1.0 - s) * mu0 + s * mu1
        out[i] = (ca0 - mu0) @ Rs.T + mus + s * residual
    return out
''',
    "02_screw_ease_residual": SHARED + '''
def predict_interior(ca_start, ca_end, fracs, mask):
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    L = ca0.shape[0]
    Ti = len(fracs)
    out = np.empty((Ti, L, 3), dtype=np.float64)
    if m.sum() < 3:
        for i, s in enumerate(fracs):
            out[i] = (1.0 - s) * ca0 + s * ca1
        return out
    w = m.astype(np.float64)
    sw = w.sum()
    mu0 = (w[:, None] * ca0).sum(0) / sw
    mu1 = (w[:, None] * ca1).sum(0) / sw
    R = _kabsch_R(ca0[m], ca1[m])
    omega = _so3_log(R)
    rigid_end = (ca0 - mu0) @ R.T + mu1
    residual = ca1 - rigid_end
    for i, s in enumerate(fracs):
        s = float(s)
        # smoothstep ease for rotation; residual stays linear in s
        u = s * s * (3.0 - 2.0 * s)
        Rs = _so3_exp(omega * u)
        mus = (1.0 - s) * mu0 + s * mu1
        out[i] = (ca0 - mu0) @ Rs.T + mus + s * residual
    return out
''',
    "03_multiseg_screw": SHARED + '''
def predict_interior(ca_start, ca_end, fracs, mask):
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    L = ca0.shape[0]
    Ti = len(fracs)
    out = np.empty((Ti, L, 3), dtype=np.float64)
    idx = np.where(m)[0]
    if idx.size < 6:
        for i, s in enumerate(fracs):
            out[i] = (1.0 - s) * ca0 + s * ca1
        return out
    # 3 equal segments over masked chain order
    nseg = 3
    cuts = np.linspace(0, idx.size, nseg + 1).astype(int)
    segs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 3:
            continue
        sel = idx[a:b]
        R = _kabsch_R(ca0[sel], ca1[sel])
        w = np.ones(sel.size)
        mu0 = ca0[sel].mean(0)
        mu1 = ca1[sel].mean(0)
        omega = _so3_log(R)
        rigid_end = (ca0[sel] - mu0) @ R.T + mu1
        residual = ca1[sel] - rigid_end
        segs.append((sel, mu0, mu1, omega, residual))
    # global residual fallback
    w = m.astype(np.float64)
    sw = w.sum()
    gmu0 = (w[:, None] * ca0).sum(0) / sw
    gmu1 = (w[:, None] * ca1).sum(0) / sw
    gR = _kabsch_R(ca0[m], ca1[m])
    gomega = _so3_log(gR)
    gres = ca1 - ((ca0 - gmu0) @ gR.T + gmu1)
    for i, s in enumerate(fracs):
        s = float(s)
        frame = (1.0 - s) * ca0 + s * ca1  # base
        # overwrite with multi-seg screw where defined
        for sel, mu0, mu1, omega, residual in segs:
            Rs = _so3_exp(omega * s)
            mus = (1.0 - s) * mu0 + s * mu1
            frame[sel] = (ca0[sel] - mu0) @ Rs.T + mus + s * residual
        # unmarked residues: global screw+res
        marked = np.zeros(L, dtype=bool)
        for sel, *_ in segs:
            marked[sel] = True
        un = m & ~marked
        if un.any():
            Rs = _so3_exp(gomega * s)
            mus = (1.0 - s) * gmu0 + s * gmu1
            frame[un] = (ca0[un] - gmu0) @ Rs.T + mus + s * gres[un]
        out[i] = frame
    return out
''',
    "04_screw_bondfix": SHARED + '''
def predict_interior(ca_start, ca_end, fracs, mask):
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    L = ca0.shape[0]
    Ti = len(fracs)
    out = np.empty((Ti, L, 3), dtype=np.float64)
    if m.sum() < 3:
        for i, s in enumerate(fracs):
            out[i] = (1.0 - s) * ca0 + s * ca1
        return out
    w = m.astype(np.float64)
    sw = w.sum()
    mu0 = (w[:, None] * ca0).sum(0) / sw
    mu1 = (w[:, None] * ca1).sum(0) / sw
    R = _kabsch_R(ca0[m], ca1[m])
    omega = _so3_log(R)
    residual = ca1 - ((ca0 - mu0) @ R.T + mu1)
    for i, s in enumerate(fracs):
        s = float(s)
        Rs = _so3_exp(omega * s)
        mus = (1.0 - s) * mu0 + s * mu1
        frame = (ca0 - mu0) @ Rs.T + mus + s * residual
        out[i] = _restore_ca_bonds(frame, m, target=3.8, strength=0.4)
    return out
''',
    "05_motion_weighted_blend": SHARED + '''
def predict_interior(ca_start, ca_end, fracs, mask):
    """Per-residue blend between linear and local rigid screw based on motion size."""
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    L = ca0.shape[0]
    Ti = len(fracs)
    out = np.empty((Ti, L, 3), dtype=np.float64)
    if m.sum() < 3:
        for i, s in enumerate(fracs):
            out[i] = (1.0 - s) * ca0 + s * ca1
        return out
    disp = np.linalg.norm(ca1 - ca0, axis=1)
    # local windows for quasi-rigid estimate
    half = 6
    screw = np.zeros((Ti, L, 3), dtype=np.float64)
    for i in range(L):
        lo = max(0, i - half)
        hi = min(L, i + half + 1)
        sel = np.arange(lo, hi)
        sel = sel[m[sel]]
        if sel.size < 3:
            for t, s in enumerate(fracs):
                screw[t, i] = (1.0 - s) * ca0[i] + s * ca1[i]
            continue
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0 = ca0[sel].mean(0)
        mu1 = ca1[sel].mean(0)
        omega = _so3_log(R)
        res_i = ca1[i] - ((ca0[i] - mu0) @ R.T + mu1)
        for t, s in enumerate(fracs):
            s = float(s)
            Rs = _so3_exp(omega * s)
            mus = (1.0 - s) * mu0 + s * mu1
            screw[t, i] = (ca0[i] - mu0) @ Rs.T + mus + s * res_i
    # blend weight: more screw where displacement is large
    d = disp.copy()
    d[~m] = 0.0
    med = float(np.median(d[m])) + 1e-8
    alpha = np.clip(d / (2.0 * med), 0.0, 1.0)  # (L,)
    for t, s in enumerate(fracs):
        s = float(s)
        lin = (1.0 - s) * ca0 + s * ca1
        a = alpha[:, None]
        out[t] = (1.0 - a) * lin + a * screw[t]
    return out
''',
    "06_hinge_two_domain": SHARED + '''
def predict_interior(ca_start, ca_end, fracs, mask):
    """Split chain at residue of max bending; two-domain screws + blend."""
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    L = ca0.shape[0]
    Ti = len(fracs)
    out = np.empty((Ti, L, 3), dtype=np.float64)
    idx = np.where(m)[0]
    if idx.size < 10:
        for i, s in enumerate(fracs):
            out[i] = (1.0 - s) * ca0 + s * ca1
        return out
    # hinge = argmax local displacement gradient along chain
    d = np.linalg.norm(ca1 - ca0, axis=1)
    grad = np.zeros(L)
    for a, b in zip(idx[:-1], idx[1:]):
        if b == a + 1:
            grad[b] = abs(d[b] - d[a])
    hinge = int(np.argmax(grad)) if grad.max() > 0 else int(idx[idx.size // 2])
    # ensure hinge interior
    hinge = int(np.clip(hinge, idx[3], idx[-4]))
    left = idx[idx < hinge]
    right = idx[idx >= hinge]
    domains = []
    for sel in (left, right):
        if sel.size < 3:
            continue
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0 = ca0[sel].mean(0)
        mu1 = ca1[sel].mean(0)
        omega = _so3_log(R)
        residual = ca1[sel] - ((ca0[sel] - mu0) @ R.T + mu1)
        domains.append((sel, mu0, mu1, omega, residual))
    for i, s in enumerate(fracs):
        s = float(s)
        frame = (1.0 - s) * ca0 + s * ca1
        for sel, mu0, mu1, omega, residual in domains:
            Rs = _so3_exp(omega * s)
            mus = (1.0 - s) * mu0 + s * mu1
            frame[sel] = (ca0[sel] - mu0) @ Rs.T + mus + s * residual
        # smooth near hinge
        for j in range(max(0, hinge - 2), min(L, hinge + 3)):
            if not m[j]:
                continue
            frame[j] = 0.5 * frame[j] + 0.5 * ((1.0 - s) * ca0[j] + s * ca1[j])
        out[i] = frame
    return out
''',
    "07_ensemble_vote": SHARED + '''
def predict_interior(ca_start, ca_end, fracs, mask):
    """Average linear, global screw+res, and 3-seg screw (ensemble)."""
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    L = ca0.shape[0]
    Ti = len(fracs)
    lin = np.empty((Ti, L, 3), dtype=np.float64)
    for i, s in enumerate(fracs):
        lin[i] = (1.0 - s) * ca0 + s * ca1
    if m.sum() < 3:
        return lin
    w = m.astype(np.float64)
    sw = w.sum()
    mu0 = (w[:, None] * ca0).sum(0) / sw
    mu1 = (w[:, None] * ca1).sum(0) / sw
    R = _kabsch_R(ca0[m], ca1[m])
    omega = _so3_log(R)
    residual = ca1 - ((ca0 - mu0) @ R.T + mu1)
    screw = np.empty_like(lin)
    for i, s in enumerate(fracs):
        s = float(s)
        Rs = _so3_exp(omega * s)
        mus = (1.0 - s) * mu0 + s * mu1
        screw[i] = (ca0 - mu0) @ Rs.T + mus + s * residual
    # 3-seg
    idx = np.where(m)[0]
    nseg = 3
    cuts = np.linspace(0, idx.size, nseg + 1).astype(int)
    multi = lin.copy()
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 3:
            continue
        sel = idx[a:b]
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0s = ca0[sel].mean(0)
        mu1s = ca1[sel].mean(0)
        om = _so3_log(R)
        res = ca1[sel] - ((ca0[sel] - mu0s) @ R.T + mu1s)
        for i, s in enumerate(fracs):
            s = float(s)
            Rs = _so3_exp(om * s)
            mus = (1.0 - s) * mu0s + s * mu1s
            multi[i, sel] = (ca0[sel] - mu0s) @ Rs.T + mus + s * res
    # weighted: favor multi/screw slightly over pure linear
    return 0.25 * lin + 0.35 * screw + 0.40 * multi
''',
    # --- gen2: local / non-rigid focused ---
    "08_pca_mode_interp": SHARED + '''
def predict_interior(ca_start, ca_end, fracs, mask):
    """Interpolate along the single dominant displacement mode (SVD of endpoint delta)."""
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    L = ca0.shape[0]
    Ti = len(fracs)
    out = np.empty((Ti, L, 3), dtype=np.float64)
    d = (ca1 - ca0).reshape(-1)
    w = np.repeat(m.astype(np.float64), 3)
    d_w = d * w
    n = float(np.linalg.norm(d_w)) + 1e-12
    mode = d_w / n
    amp = float((d * w * mode).sum())  # projection
    for i, s in enumerate(fracs):
        s = float(s)
        # nonlinear mid-path boost of mode (sine half-wave) + linear residual
        mid = np.sin(np.pi * s)
        delta = (s * amp * mode + 0.15 * mid * amp * mode).reshape(L, 3)
        out[i] = ca0 + delta
        # mix back residual that mode misses
        full = s * (ca1 - ca0)
        miss = full - (s * amp * mode).reshape(L, 3)
        out[i] = out[i] + miss
    return out
''',
    "09_cubic_hermite_endpoints": SHARED + '''
def predict_interior(ca_start, ca_end, fracs, mask):
    """Cubic Hermite with zero endpoint velocities (ease-in/out path, non-uniform speed)."""
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    out = np.empty((len(fracs), ca0.shape[0], 3), dtype=np.float64)
    for i, s in enumerate(fracs):
        s = float(s)
        # Hermite basis with v0=v1=0: h00=2s^3-3s^2+1, h01=-2s^3+3s^2
        h00 = 2 * s**3 - 3 * s**2 + 1
        h01 = -2 * s**3 + 3 * s**2
        out[i] = h00 * ca0 + h01 * ca1
    return out
''',
    "10_anisotropic_residual": SHARED + '''
def predict_interior(ca_start, ca_end, fracs, mask):
    """Global screw + per-residue residual with mid-path anisotropic boost."""
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    L = ca0.shape[0]
    Ti = len(fracs)
    out = np.empty((Ti, L, 3), dtype=np.float64)
    if m.sum() < 3:
        for i, s in enumerate(fracs):
            out[i] = (1.0 - s) * ca0 + s * ca1
        return out
    w = m.astype(np.float64)
    sw = w.sum()
    mu0 = (w[:, None] * ca0).sum(0) / sw
    mu1 = (w[:, None] * ca1).sum(0) / sw
    R = _kabsch_R(ca0[m], ca1[m])
    omega = _so3_log(R)
    residual = ca1 - ((ca0 - mu0) @ R.T + mu1)
    # boost residues with large residual at mid-path (non-rigid cores)
    rmag = np.linalg.norm(residual, axis=1)
    rmag = rmag / (rmag.max() + 1e-8)
    for i, s in enumerate(fracs):
        s = float(s)
        Rs = _so3_exp(omega * s)
        mus = (1.0 - s) * mu0 + s * mu1
        rigid = (ca0 - mu0) @ Rs.T + mus
        # residual progress: s + beta * sin(pi s) * rmag
        beta = 0.25
        prog = s + beta * np.sin(np.pi * s) * rmag
        prog = prog[:, None]
        out[i] = rigid + prog * residual
    return out
''',
    "11_five_seg_adaptive": SHARED + '''
def predict_interior(ca_start, ca_end, fracs, mask):
    """5 segments cut by displacement quantiles (adaptive domain decomposition)."""
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    L = ca0.shape[0]
    Ti = len(fracs)
    out = np.empty((Ti, L, 3), dtype=np.float64)
    idx = np.where(m)[0]
    if idx.size < 15:
        for i, s in enumerate(fracs):
            out[i] = (1.0 - s) * ca0 + s * ca1
        return out
    d = np.linalg.norm(ca1 - ca0, axis=1)
    # order residues by chain, split into 5 by equal count (stable)
    nseg = 5
    cuts = np.linspace(0, idx.size, nseg + 1).astype(int)
    segs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 3:
            continue
        sel = idx[a:b]
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0, mu1 = ca0[sel].mean(0), ca1[sel].mean(0)
        omega = _so3_log(R)
        residual = ca1[sel] - ((ca0[sel] - mu0) @ R.T + mu1)
        segs.append((sel, mu0, mu1, omega, residual))
    for i, s in enumerate(fracs):
        s = float(s)
        frame = (1.0 - s) * ca0 + s * ca1
        for sel, mu0, mu1, omega, residual in segs:
            Rs = _so3_exp(omega * s)
            mus = (1.0 - s) * mu0 + s * mu1
            frame[sel] = (ca0[sel] - mu0) @ Rs.T + mus + s * residual
        out[i] = frame
    return out
''',
}


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    for name, body in CANDIDATES.items():
        path = HERE / f"{name}.py"
        path.write_text(HEADER + body + FOOTER, encoding="utf-8")
        print("wrote", path.name)


if __name__ == "__main__":
    main()
