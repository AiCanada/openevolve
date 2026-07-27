"""Motion-correlation domains: cluster by displacement cosine, rigid screw per domain."""
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


def _cluster_chain(disp, m, n_dom=3, min_size=8):
    """Greedy sequential domains by displacement direction breaks."""
    idx = np.where(m)[0]
    if idx.size < n_dom * min_size:
        return [idx] if idx.size else []
    # unit displacement directions
    norms = np.linalg.norm(disp, axis=1) + 1e-12
    u = disp / norms[:, None]
    # chain cosine between consecutive masked residues
    breaks = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            breaks.append((a + 0.5, 2.0))  # chain gap = strong break
            continue
        c = float(np.dot(u[a], u[b]))
        mag = 0.5 * (norms[a] + norms[b])
        score = (1.0 - c) * mag  # large when directions disagree and motion is big
        breaks.append((a + 0.5, score))
    # pick top n_dom-1 break positions with spacing
    breaks_sorted = sorted(breaks, key=lambda x: -x[1])
    chosen = []
    for pos, sc in breaks_sorted:
        if sc < 0.15:
            break
        if all(abs(pos - c) > min_size for c in chosen):
            chosen.append(pos)
        if len(chosen) >= n_dom - 1:
            break
    cuts = sorted(int(np.ceil(c)) for c in chosen)
    domains = []
    start = int(idx[0])
    end = int(idx[-1]) + 1
    bounds = [start] + cuts + [end]
    for a, b in zip(bounds[:-1], bounds[1:]):
        sel = np.arange(a, b)
        sel = sel[m[sel]]
        if sel.size >= 3:
            domains.append(sel)
    if not domains:
        domains = [idx]
    return domains


def predict_interior(ca_start, ca_end, fracs, mask):
    ca0 = np.asarray(ca_start, float)
    ca1 = np.asarray(ca_end, float)
    m = np.asarray(mask, bool)
    fracs = np.asarray(fracs, float)
    L = ca0.shape[0]
    Ti = len(fracs)
    disp = ca1 - ca0

    # global baseline
    if m.sum() >= 3:
        R = _kabsch_R(ca0[m], ca1[m])
        mu0, mu1 = ca0[m].mean(0), ca1[m].mean(0)
        om = _so3_log(R)
        res_g = ca1 - ((ca0 - mu0) @ R.T + mu1)
        glob = np.empty((Ti, L, 3))
        for i, s in enumerate(fracs):
            s = float(s)
            Rs = _so3_exp(om * s)
            mus = (1 - s) * mu0 + s * mu1
            glob[i] = (ca0 - mu0) @ Rs.T + mus + s * res_g
    else:
        glob = np.stack([(1 - s) * ca0 + s * ca1 for s in fracs], 0)

    domains = _cluster_chain(disp, m, n_dom=3, min_size=8)
    dom = glob.copy()
    for sel in domains:
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0, mu1 = ca0[sel].mean(0), ca1[sel].mean(0)
        om = _so3_log(R)
        res = ca1[sel] - ((ca0[sel] - mu0) @ R.T + mu1)
        for i, s in enumerate(fracs):
            s = float(s)
            Rs = _so3_exp(om * s)
            mus = (1 - s) * mu0 + s * mu1
            dom[i, sel] = (ca0[sel] - mu0) @ Rs.T + mus + s * res

    # light blend with global for stability
    return 0.35 * glob + 0.65 * dom
# EVOLVE-BLOCK-END


def run_interpolation(ca_start, ca_end, fracs, mask):
    return predict_interior(ca_start, ca_end, fracs, mask)
