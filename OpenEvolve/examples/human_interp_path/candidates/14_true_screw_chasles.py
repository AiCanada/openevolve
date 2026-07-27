"""True Chasles screw (rotation about fixed axis + axial slide) + residual."""
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


def _chasles_point(mu0, mu1, R, omega):
    """Point on screw axis and pitch for rigid body from (mu0,R)->mu1."""
    theta = float(np.linalg.norm(omega))
    t = mu1 - mu0 @ R.T  # translation after rotating about origin? use centered form
    # Prefer: t = mu1 - R @ mu0 when points are absolute under R about origin
    t = mu1 - (R @ mu0)
    if theta < 1e-8:
        return mu0, t  # pure translation: axis through mu0, "pitch" vector = t
    k = omega / theta
    # Decompose t into parallel + perpendicular to axis
    t_par = float(np.dot(t, k)) * k
    t_perp = t - t_par
    # Point on axis: p = 0.5 * t_perp + (k x t_perp) / (2 tan(theta/2))  (Rodrigues)
    # For rotation about p: x' = R(x-p) + p + t_par_scaled
    # Closed form for screw axis point from rigid motion (R,t):
    # p = 0.5 * (t - (k·t)*k) + (k × t) / (2 tan(θ/2))
    tan_half = np.tan(theta * 0.5)
    if abs(tan_half) < 1e-10:
        p = mu0
    else:
        p = 0.5 * t_perp + np.cross(k, t) / (2.0 * tan_half)
    return p, t_par


def predict_interior(ca_start, ca_end, fracs, mask):
    ca0 = np.asarray(ca_start, float)
    ca1 = np.asarray(ca_end, float)
    m = np.asarray(mask, bool)
    fracs = np.asarray(fracs, float)
    L = ca0.shape[0]
    Ti = len(fracs)
    out = np.empty((Ti, L, 3))
    if m.sum() < 3:
        for i, s in enumerate(fracs):
            out[i] = (1 - s) * ca0 + s * ca1
        return out

    R = _kabsch_R(ca0[m], ca1[m])
    mu0 = ca0[m].mean(0)
    mu1 = ca1[m].mean(0)
    omega = _so3_log(R)
    # Standard form: x1 ≈ R @ (x0 - mu0) + mu1  ⇒ effective t about origin:
    # x1 = R @ x0 + t with t = mu1 - R @ mu0
    t = mu1 - R @ mu0
    theta = float(np.linalg.norm(omega))
    if theta < 1e-8:
        # pure translation + residual
        residual = ca1 - (ca0 + t)
        for i, s in enumerate(fracs):
            s = float(s)
            out[i] = ca0 + s * t + s * residual
        return out

    k = omega / theta
    t_par = float(np.dot(t, k)) * k
    t_perp = t - t_par
    tan_half = np.tan(theta * 0.5)
    if abs(tan_half) < 1e-10:
        p = mu0
    else:
        p = 0.5 * t_perp + np.cross(k, t) / (2.0 * tan_half)

    # rigid endpoint via full screw should match Kabsch rigid
    rigid_end = (ca0 - p) @ R.T + p + t_par
    residual = ca1 - rigid_end

    for i, s in enumerate(fracs):
        s = float(s)
        Rs = _so3_exp(omega * s)
        rigid = (ca0 - p) @ Rs.T + p + s * t_par
        out[i] = rigid + s * residual
    return out
# EVOLVE-BLOCK-END


def run_interpolation(ca_start, ca_end, fracs, mask):
    return predict_interior(ca_start, ca_end, fracs, mask)
