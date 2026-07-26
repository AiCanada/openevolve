"""Transition-path interpolation for protein backbones (Project H.U.M.A.N.).

TASK
----
Given the Cα coordinates of the TWO ENDPOINT conformations of a molecular-dynamics
window, predict the Cα coordinates of the interior frames along the transition path.

This is scored exactly as `scripts/eval_interp_path.py` Metric A scores it: per-frame
Cα Kabsch RMSD against the true MD frames, on mid-path frames (fraction in [0.35, 0.65]),
against the naive straight-line baseline that the project must beat.

Because the score uses Kabsch (optimal rigid) superposition, any global rotation or
translation you apply is removed for free — only the INTERNAL geometry of your predicted
conformation matters. Improvements must come from modelling how the protein actually
moves between the two endpoints, not from repositioning it.

Physical facts that may be useful:
  * Consecutive Cα atoms sit ~3.8 Å apart. Straight-line interpolation between two
    conformations SHORTENS these distances wherever the chain rotates, producing
    physically impossible compressed intermediates.
  * A rotating rigid body traces an ARC, but linear interpolation traces the CHORD,
    which cuts the corner and under-shoots the true radius.
  * Proteins move largely as quasi-rigid groups (domains/secondary structure) hinging
    about a few flexible joints, rather than every atom moving independently.
  * The true interior frames are MD samples, so they also contain thermal fluctuation
    that no deterministic predictor can reproduce; there is an irreducible error floor.
"""

import numpy as np

# EVOLVE-BLOCK-START


def _kabsch_R(P, Q, w=None):
    """Rotation R minimising ||R P - Q|| (column vectors: R @ p). P,Q (n,3)."""
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
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R


def _so3_log(R):
    """so(3) log map: rotation matrix -> axis-angle vector (3,)."""
    cos_t = (np.trace(R) - 1.0) * 0.5
    cos_t = float(np.clip(cos_t, -1.0, 1.0))
    theta = float(np.arccos(cos_t))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if abs(np.pi - theta) < 1e-6:
        # Near 180°: recover axis from diagonal of R
        axis = np.sqrt(np.maximum(np.diag(R) + 1.0, 0.0))
        if axis.sum() < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            # sign from off-diagonals
            if R[2, 1] < R[1, 2]:
                axis[0] = -axis[0]
            if R[0, 2] < R[2, 0]:
                axis[1] = -axis[1]
            if R[1, 0] < R[0, 1]:
                axis[2] = -axis[2]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    w = w * (theta / (2.0 * np.sin(theta) + 1e-12))
    return w


def _so3_exp(w):
    """so(3) exp map: axis-angle vector -> rotation matrix."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-10:
        return np.eye(3)
    k = w / theta
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def predict_interior(ca_start, ca_end, fracs, mask):
    """Screw / rigid-body SLERP + residual linear blend.

    1) Fit Kabsch rotation+translation mapping start -> end on masked residues.
    2) Interpolate the rigid motion along SO(3)×R^3 (arc, not chord).
    3) Add a residual that is linear in the non-rigid deformation so local
       flexibility is not forced to pure rigidity.
    """
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
    R = _kabsch_R(ca0[m], ca1[m])  # R @ (p-mu0) + mu1 ≈ q
    omega = _so3_log(R)

    # Non-rigid residual at end: ca1 - rigid_image(ca0)
    rigid_end = (ca0 - mu0) @ R.T + mu1
    residual = ca1 - rigid_end

    for i, s in enumerate(fracs):
        s = float(s)
        Rs = _so3_exp(omega * s)
        mus = (1.0 - s) * mu0 + s * mu1
        rigid_s = (ca0 - mu0) @ Rs.T + mus
        # blend residual (linear non-rigid deformation)
        out[i] = rigid_s + s * residual
    return out


# EVOLVE-BLOCK-END


def run_interpolation(ca_start, ca_end, fracs, mask):
    """Fixed entry point used by the evaluator."""
    return predict_interior(ca_start, ca_end, fracs, mask)
