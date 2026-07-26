"""Transition-path interpolation for protein backbones (Project H.U.M.A.N.).

TASK
----
Given the CA coordinates of the TWO ENDPOINT conformations of a molecular-dynamics
window, predict the CA coordinates of the interior frames along the transition path.
Scored via per-frame CA Kabsch RMSD against true MD frames on mid-path frames
(fraction in [0.35, 0.65]) versus a straight-line baseline, using the PAIRED median
relative improvement across windows.

MEASURED FACTS ON THIS DATASET
-------------------------------
1. The global Kabsch rotation between endpoints has a median of ~0.05 degrees, and
   the scorer re-superposes every frame with Kabsch before computing RMSD. A single
   GLOBAL rigid screw motion is a safe, endpoint-exact scaffold, but essentially a
   no-op predictor here -- it cannot be the source of any real gain.
2. Many internal-motion mechanisms tried in prior evolutionary search on this exact
   split regress BELOW the linear baseline when trusted as the dominant term: CA-CA
   bond restoration, distance-matrix/MDS reconstruction, ungated per-residue rigid
   fits, and multi-segment LBS. The common failure mode is that any locally-scoped
   fit from only two static (thermally-noisy) snapshots is dominated by sampling
   noise, and confidently extrapolating that noise to the mid-path fraction adds
   uncorrelated variance faster than it removes the (small) corner-cutting bias.
3. The best verified configuration so far (combined_score +0.0015, win_rate 0.622,
   sign_test_p 0.027) is a heavily shrunk ENSEMBLE: a near-linear global screw term,
   a reliability-gated multi-scale (fine/mid/coarse, Gaussian-in-sequence-index)
   local rigid field, and a rank-<=3 SVD "collective displacement direction" term,
   all convexly blended with small, conservative weights.

THIS VERSION -- what's new
---------------------------
The previous "mode" term decomposes the raw endpoint displacement field D (masked
residues x 3) via its own SVD. Because D only has 3 *columns*, that SVD is
intrinsically rank <= 3 and the basis it uses is derived purely from the data being
fit (the displacement itself) -- it has no notion of the protein's actual 3D
topology, so it can only ever represent "a handful of global Cartesian sweep
directions", not genuinely multi-domain collective motion.

We add one new, structurally-motivated term, `enm`: a topology-informed smooth
low-rank basis built the way Elastic/Gaussian Network Models build collective
modes -- a weighted residue-residue coupling graph from `ca0` geometry (soft
inverse-distance-squared spring weights, no hard cutoff so the graph stays
connected and well-conditioned), whose lowest-frequency Kirchhoff/Laplacian
eigenvectors are exactly the "soft, extended" motions real quasi-rigid domains
tend to follow. We fit the *observed* displacement D as a least-squares
combination of the top-k such eigenvectors (k ~ 6, still tiny vs. L) and treat the
unexplained remainder as a purely linear residual. Unlike the previous mode term,
the basis here comes from `ca0`'s structure, not from D's own SVD, so it can
represent multi-domain, sequence-nonlocal coupling patterns that a rank-3 Cartesian
SVD structurally cannot.

Weight bookkeeping: the new `w_enm` slice is carved out of `w_screw` (0.62 -> 0.56),
since fact (1) establishes `screw` contributes essentially nothing beyond `lin` on
this data -- reallocating from a proven no-op term is the lowest-risk way to fund a
new mechanism. `w_lin`, `w_local`, `w_mode` are untouched from the verified-good
configuration.

Safety
------
Every term (`lin`, `screw`, `local`, `mode`, `enm`) reproduces the two given
endpoints EXACTLY (s=0 -> ca0, s=1 -> ca1), so the convex blend does too, for any
weights and regardless of numerical noise in any individual fit. `enm` additionally
degrades to `lin` gracefully (not just at the endpoints) whenever the basis can't be
built (too few residues, singular fit, non-finite values) -- see `_enm_mode_path`.
Every internal-motion term is wrapped so a failure degrades that one term to `lin`
rather than propagating NaNs/Infs, and the public entry point has a hard fallback to
plain linear interpolation on any unexpected failure.

All operations are vectorised over residues (batched 3x3 SVD / exp / log maps for
the local rigid field; one dense LxL eigh for the ENM basis, cheap and deterministic
for L <= 512, no iterative/randomized solver involved).
"""

import numpy as np

# EVOLVE-BLOCK-START

PARAMS = {
    # convex-blend weights (renormalised to sum to 1; any may be 0). w_enm is
    # newly carved out of w_screw (0.62 -> 0.56); w_lin/w_local/w_mode are
    # unchanged from the verified-good configuration.
    "w_lin": 0.13,
    "w_screw": 0.56,
    "w_local": 0.17,
    "w_mode": 0.08,
    "w_enm": 0.06,
    # optional non-uniform time reparameterisation (0 => identity, i.e. plain s).
    "ease": 0.0,
    "hermite": 0.0,
    # multi-scale local-rigid-field gating (fine/mid/coarse): verified-good values.
    "trust_cap": 0.55,
    "fine_r2_lo": 0.95,
    "fine_r2_hi": 0.99,
    "fine_supp_lo": 10.0,
    "fine_supp_hi": 20.0,
    "mid_r2_lo": 0.93,
    "mid_r2_hi": 0.98,
    "mid_supp_lo": 16.0,
    "mid_supp_hi": 30.0,
    "coarse_r2_lo": 0.90,
    "coarse_r2_hi": 0.97,
    "coarse_supp_lo": 25.0,
    "coarse_supp_hi": 50.0,
    # low-rank collective-mode term (Cartesian-direction SVD of the raw
    # displacement field).
    "n_modes": 1,
    "mode_mid_boost": 0.12,
    # NEW: topology-informed ENM/GNM-style low-rank displacement basis.
    "enm_k": 6,
    "enm_gamma_power": 2.0,
    "enm_mid_boost": 0.08,
}


# ----------------------------------------------------------------------------
# Scalar SO(3) helpers (used for the single global screw fit)
# ----------------------------------------------------------------------------


def _kabsch_R(P, Q):
    """Rotation R minimising ||P @ R.T - Q||_F. P, Q: (n,3), n >= 3."""
    if P.shape[0] < 3:
        return np.eye(3)
    mu_p, mu_q = P.mean(0), Q.mean(0)
    H = (P - mu_p).T @ (Q - mu_q)
    if not np.all(np.isfinite(H)):
        return np.eye(3)
    try:
        U, _S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return np.eye(3)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    if d == 0:
        d = 1.0
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def _so3_log_scalar(R):
    """so(3) log map: rotation matrix -> axis-angle vector (3,)."""
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


def _so3_exp_scalar(w):
    """so(3) exp map: axis-angle vector (3,) -> rotation matrix (3,3)."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-10:
        return np.eye(3)
    k = w / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _time(s, ease, hermite):
    """Blend of linear / smoothstep / cubic-Hermite timing; identity at ease=hermite=0."""
    s = float(s)
    h = float(np.clip(hermite, 0.0, 1.0))
    e = float(np.clip(ease, 0.0, 1.0))
    sh = 3.0 * s * s - 2.0 * s * s * s
    se = s * s * (3.0 - 2.0 * s)
    t = (1.0 - h) * ((1.0 - e) * s + e * se) + h * sh
    return float(np.clip(t, 0.0, 1.0))


def _global_screw_path(ca0, ca1, sel, fracs, ease, herm):
    """Fit ONE rigid screw motion (Kabsch) on `sel`, interpolate along SO(3).

    Leftover non-rigid residual is blended in linearly with the (re-timed)
    fraction, so the two given endpoints are always reproduced exactly.
    Returns an array (Ti, len(sel), 3).
    """
    Ti = len(fracs)
    out = np.empty((Ti, sel.size, 3), dtype=np.float64)
    R = _kabsch_R(ca0[sel], ca1[sel])
    mu0 = ca0[sel].mean(0)
    mu1 = ca1[sel].mean(0)
    om = _so3_log_scalar(R)
    d0 = ca0[sel] - mu0
    rigid_end = d0 @ R.T + mu1
    residual = ca1[sel] - rigid_end
    for i, s in enumerate(fracs):
        t = _time(s, ease, herm)
        Rs = _so3_exp_scalar(om * t)
        mus = (1.0 - t) * mu0 + t * mu1
        out[i] = d0 @ Rs.T + mus + t * residual
    return out


# ----------------------------------------------------------------------------
# Batched SO(3) helpers (used for the multi-scale local rigid field)
# ----------------------------------------------------------------------------


def _so3_log_batch(R):
    """Vectorised so(3) log map. R: (N,3,3) -> (N,3) axis-angle vectors."""
    tr = np.einsum("nii->n", R)
    cos_t = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_t)
    w = np.stack(
        [R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]],
        axis=1,
    )
    small = theta < 1e-8
    denom = 2.0 * np.sin(theta)
    denom_safe = np.where(small, 1.0, denom)
    scale = np.where(small, 0.0, theta / denom_safe)
    out = w * scale[:, None]

    near_pi = np.abs(np.pi - theta) < 1e-6
    if np.any(near_pi):
        for i in np.where(near_pi)[0]:
            out[i] = _so3_log_scalar(R[i])
    return out


def _so3_exp_batch(W):
    """Vectorised so(3) exp map. W: (N,3) axis-angle vectors -> (N,3,3) rotations."""
    N = W.shape[0]
    theta = np.linalg.norm(W, axis=1)
    small = theta < 1e-10
    theta_safe = np.where(small, 1.0, theta)
    k = W / theta_safe[:, None]

    K = np.zeros((N, 3, 3), dtype=np.float64)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]

    K2 = np.einsum("nij,njk->nik", K, K)
    I = np.eye(3, dtype=np.float64)[None, :, :]
    sin_t = np.sin(theta)[:, None, None]
    cos_t = (1.0 - np.cos(theta))[:, None, None]
    R = I + sin_t * K + cos_t * K2
    R[small] = np.eye(3, dtype=np.float64)
    return R


def _smoothstep(x, lo, hi):
    t = (x - lo) / max(hi - lo, 1e-12)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _local_rigid_fit(ca0, ca1, m, sigma):
    """Gaussian-in-sequence-index weighted Kabsch fit, batched over all residues.

    Returns, for every residue i: weighted local centroids mu0_i, mu1_i, the
    best-fit local rotation R_i (start -> end), the closed-form R^2 of that fit
    (fraction of local positional variance explained by the rigid rotation), and
    the effective (weighted) support size sw_i.
    """
    L = ca0.shape[0]
    idx = np.arange(L, dtype=np.float64)
    dm = idx[:, None] - idx[None, :]
    mf = m.astype(np.float64)

    W = np.exp(-0.5 * (dm / sigma) ** 2) * mf[None, :]
    sw = W.sum(1)
    sw_safe = np.maximum(sw, 1e-8)

    mu0 = (W @ ca0) / sw_safe[:, None]
    mu1 = (W @ ca1) / sw_safe[:, None]

    outer_flat = (ca0[:, :, None] * ca1[:, None, :]).reshape(L, 9)
    A = W @ outer_flat
    H = A.reshape(L, 3, 3) - sw_safe[:, None, None] * (mu0[:, :, None] * mu1[:, None, :])

    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    d = np.where(d == 0, 1.0, d)
    diagm = np.zeros((L, 3, 3), dtype=np.float64)
    diagm[:, 0, 0] = 1.0
    diagm[:, 1, 1] = 1.0
    diagm[:, 2, 2] = d

    VtT = np.transpose(Vt, (0, 2, 1))
    UT = np.transpose(U, (0, 2, 1))
    R = np.einsum("nij,njk,nkl->nil", VtT, diagm, UT)

    sq0 = np.einsum("ij,ij->i", ca0, ca0)
    sq1 = np.einsum("ij,ij->i", ca1, ca1)
    sumX2 = np.maximum((W @ sq0) - sw_safe * np.einsum("ij,ij->i", mu0, mu0), 0.0)
    sumY2 = np.maximum((W @ sq1) - sw_safe * np.einsum("ij,ij->i", mu1, mu1), 0.0)

    ss_res = np.maximum(sumX2 + sumY2 - 2.0 * (S[:, 0] + S[:, 1] + d * S[:, 2]), 0.0)
    ss_tot = 0.5 * (sumX2 + sumY2)
    floor_var = sw_safe * 0.09  # ~0.3 A minimal per-residue spread (numerical floor)
    ss_tot_safe = np.maximum(ss_tot, floor_var)
    r2 = np.clip(1.0 - ss_res / np.maximum(ss_tot_safe, 1e-8), 0.0, 1.0)

    return mu0, mu1, R, r2, sw


def _scale_configs(L, p):
    """Three Gaussian-window (sigma, r2_lo, r2_hi, supp_lo, supp_hi) scale specs
    spanning turn / secondary-structure / small-domain sequence-neighbourhood
    sizes -- the verified-good values, unchanged.
    """
    sigma_fine = float(np.clip(L / 25.0, 5.0, 12.0))
    sigma_mid = float(np.clip(L / 10.0, 13.0, 22.0))
    sigma_coarse = float(np.clip(L / 5.0, 20.0, 60.0))
    return [
        (sigma_fine, p["fine_r2_lo"], p["fine_r2_hi"], p["fine_supp_lo"], p["fine_supp_hi"]),
        (sigma_mid, p["mid_r2_lo"], p["mid_r2_hi"], p["mid_supp_lo"], p["mid_supp_hi"]),
        (sigma_coarse, p["coarse_r2_lo"], p["coarse_r2_hi"], p["coarse_supp_lo"], p["coarse_supp_hi"]),
    ]


def _local_field_path(ca0, ca1, m, fracs, p):
    """Multi-scale (N independently gated scales) reliability-gated local rigid
    field. Endpoint-exact; degrades EXACTLY to plain linear interpolation for any
    residue where no scale finds convincing rigidity: with omega_i = 0 for every
    scale, the SO(3) exp is the identity at every s and every centroid term
    telescopes away algebraically, leaving pos_i(s) = (1-s) ca0_i + s ca1_i exactly
    -- regardless of how many scales were blended or what their centroids were.
    """
    L = ca0.shape[0]
    Ti = len(fracs)
    out = np.empty((Ti, L, 3), dtype=np.float64)

    trust_cap = float(p.get("trust_cap", 0.55))
    configs = _scale_configs(L, p)

    mu0_list, mu1_list, omega_raw_list, rel_list = [], [], [], []
    for sigma, r2_lo, r2_hi, supp_lo, supp_hi in configs:
        mu0_k, mu1_k, R_k, r2_k, sw_k = _local_rigid_fit(ca0, ca1, m, sigma)
        rel_k = trust_cap * _smoothstep(r2_k, r2_lo, r2_hi) * _smoothstep(sw_k, supp_lo, supp_hi)
        mu0_list.append(mu0_k)
        mu1_list.append(mu1_k)
        omega_raw_list.append(_so3_log_batch(R_k))
        rel_list.append(rel_k)

    rel_stack = np.stack(rel_list, axis=0)  # (n_scales, L)
    eps = 1e-9
    wsum = rel_stack.sum(0) + eps  # (L,)
    w_norm = rel_stack / wsum[None, :]  # (n_scales, L)

    mu0_sel = np.zeros((L, 3), dtype=np.float64)
    mu1_sel = np.zeros((L, 3), dtype=np.float64)
    omega = np.zeros((L, 3), dtype=np.float64)
    for k in range(len(configs)):
        wk = w_norm[k][:, None]
        mu0_sel += wk * mu0_list[k]
        mu1_sel += wk * mu1_list[k]
        omega += wk * (omega_raw_list[k] * rel_list[k][:, None])

    d0v = ca0 - mu0_sel
    R_end = _so3_exp_batch(omega)
    rigid_end = mu1_sel + np.einsum("nij,nj->ni", R_end, d0v)
    residual = ca1 - rigid_end

    ease = float(p.get("ease", 0.0))
    herm = float(p.get("hermite", 0.0))
    not_m = ~m
    for fi, s in enumerate(fracs):
        t = _time(s, ease, herm)
        Rs = _so3_exp_batch(omega * t)
        mus = (1.0 - t) * mu0_sel + t * mu1_sel
        rigid_s = mus + np.einsum("nij,nj->ni", Rs, d0v)
        pos = rigid_s + t * residual
        if np.any(not_m):
            pos[not_m] = (1.0 - t) * ca0[not_m] + t * ca1[not_m]
        out[fi] = pos
    return out


def _mode_path(ca0, ca1, m_idx, fracs, p, lin, ease, herm):
    """Low-rank (Cartesian-direction) collective-displacement term. Unmasked
    residues are always exactly `lin`.
    """
    nmodes = max(0, int(p.get("n_modes", 1)))
    if nmodes <= 0 or m_idx.size < 6:
        return lin
    D = ca1[m_idx] - ca0[m_idx]
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    k = min(nmodes, U.shape[1], 3)
    recon = np.zeros_like(D)
    for j in range(k):
        recon += np.outer(U[:, j] * S[j], Vt[j])
    miss = D - recon
    if not (np.all(np.isfinite(recon)) and np.all(np.isfinite(miss))):
        return lin
    beta = float(p.get("mode_mid_boost", 0.0))
    cand = lin.copy()
    base0 = ca0[m_idx]
    for i, s in enumerate(fracs):
        t = _time(s, ease, herm)
        mid = beta * np.sin(np.pi * float(s))
        disp = (t + mid) * recon + t * miss
        cand[i, m_idx] = base0 + disp
    if not np.all(np.isfinite(cand)):
        return lin
    return cand


# ----------------------------------------------------------------------------
# NEW: topology-informed ENM/GNM-style low-rank displacement basis
# ----------------------------------------------------------------------------


def _enm_basis(P, k, gamma_power):
    """Lowest-frequency Kirchhoff/Laplacian eigenvectors of a soft, fully-
    connected inverse-distance-power coupling graph built from `P` (n,3), a
    Gaussian/Elastic-Network-Model-style construction of collective motion
    basis vectors purely from structure (no hard cutoff, so the graph is
    guaranteed connected and the Laplacian well-conditioned).

    Returns a (n, k) array of basis vectors (columns), or None if `n` is too
    small or the eigendecomposition fails. Deterministic: a single dense
    `eigh` on an n x n symmetric matrix, no iterative/randomized solver.
    """
    n = P.shape[0]
    if n < k + 2:
        return None
    sq = np.einsum("ij,ij->i", P, P)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (P @ P.T)
    d2 = np.maximum(d2, 1e-6)
    W = 1.0 / np.power(d2, gamma_power * 0.5)
    np.fill_diagonal(W, 0.0)
    if not np.all(np.isfinite(W)):
        return None
    deg = W.sum(1)
    Gamma = np.diag(deg) - W
    try:
        evals, evecs = np.linalg.eigh(Gamma)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(evecs)):
        return None
    # evals[0] ~ 0 (the connected-graph translation/constant mode); take the
    # next k lowest-frequency non-trivial modes.
    phi = evecs[:, 1 : 1 + k]
    if phi.shape[1] < k:
        return None
    return phi


def _enm_mode_path(ca0, ca1, m_idx, fracs, p, lin, ease, herm):
    """Fit the observed masked-residue displacement as a least-squares
    combination of the top-k ENM/GNM basis vectors of `ca0`, and interpolate
    with a mild sinusoidal mid-path boost on the explained part; the
    unexplained remainder is carried linearly. Endpoint-exact and reduces
    exactly to `lin` on any failure (too few residues, singular basis/fit,
    non-finite values).
    """
    k = max(0, int(p.get("enm_k", 0)))
    n = m_idx.size
    if k <= 0 or n < k + 3:
        return lin
    P0 = ca0[m_idx]
    phi = _enm_basis(P0, k, float(p.get("enm_gamma_power", 2.0)))
    if phi is None:
        return lin
    D = ca1[m_idx] - P0
    try:
        A, _res, _rank, _sv = np.linalg.lstsq(phi, D, rcond=None)
    except np.linalg.LinAlgError:
        return lin
    recon = phi @ A
    miss = D - recon
    if not (np.all(np.isfinite(recon)) and np.all(np.isfinite(miss))):
        return lin
    beta = float(p.get("enm_mid_boost", 0.0))
    cand = lin.copy()
    for i, s in enumerate(fracs):
        t = _time(s, ease, herm)
        boost = beta * np.sin(np.pi * float(s))
        disp = (t + boost) * recon + t * miss
        cand[i, m_idx] = P0 + disp
    if not np.all(np.isfinite(cand)):
        return lin
    return cand


def _predict_interior_impl(ca0, ca1, fracs, mask):
    m = np.asarray(mask, dtype=bool)
    L = ca0.shape[0]
    Ti = len(fracs)

    if L == 0 or Ti == 0:
        return np.empty((Ti, L, 3), dtype=np.float64)

    p = PARAMS
    ease = float(p.get("ease", 0.0))
    herm = float(p.get("hermite", 0.0))

    # ---- lin: always-valid floor, exact at both endpoints ----------------
    lin = np.empty((Ti, L, 3), dtype=np.float64)
    for i, s in enumerate(fracs):
        t = _time(s, ease, herm)
        lin[i] = (1.0 - t) * ca0 + t * ca1

    if m.sum() < 3 or L < 3:
        return lin

    m_idx = np.arange(L)[m]

    # ---- screw: single global rigid fit over all masked residues ---------
    screw = lin
    try:
        cand = lin.copy()
        cand[:, m_idx] = _global_screw_path(ca0, ca1, m_idx, fracs, ease, herm)
        if np.all(np.isfinite(cand)):
            screw = cand
    except Exception:
        screw = lin

    # ---- local: multi-scale (fine/mid/coarse) reliability-gated rigid field ----
    local = lin
    if m.sum() >= 8 and L >= 8:
        try:
            cand = _local_field_path(ca0, ca1, m, fracs, p)
            if np.all(np.isfinite(cand)):
                local = cand
        except Exception:
            local = lin

    # ---- mode: low-rank Cartesian-direction collective-displacement term -----
    mode = lin
    try:
        cand = _mode_path(ca0, ca1, m_idx, fracs, p, lin, ease, herm)
        if np.all(np.isfinite(cand)):
            mode = cand
    except Exception:
        mode = lin

    # ---- enm: topology-informed ENM/GNM-style low-rank displacement basis ----
    enm = lin
    if m.sum() >= 9 and L >= 9:
        try:
            cand = _enm_mode_path(ca0, ca1, m_idx, fracs, p, lin, ease, herm)
            if np.all(np.isfinite(cand)):
                enm = cand
        except Exception:
            enm = lin

    # ---- convex blend -------------------------------------------------
    ws = np.array(
        [
            float(p["w_lin"]),
            float(p["w_screw"]),
            float(p["w_local"]),
            float(p["w_mode"]),
            float(p["w_enm"]),
        ],
        dtype=np.float64,
    )
    ws = np.maximum(ws, 0.0)
    wsum = ws.sum()
    if wsum < 1e-12:
        ws = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        wsum = 1.0
    ws = ws / wsum

    out = ws[0] * lin + ws[1] * screw + ws[2] * local + ws[3] * mode + ws[4] * enm

    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, lin)

    return out


def predict_interior(ca_start, ca_end, fracs, mask):
    """Public entry: ensemble of endpoint-exact predictors (global screw,
    reliability-gated three-scale local rigid field, low-rank Cartesian-direction
    collective mode, topology-informed ENM/GNM low-rank displacement basis, and
    plain linear as a floor), with a hard fallback to plain linear interpolation
    on ANY unexpected failure so the function is always deterministic and finite
    regardless of edge-case input (degenerate masks, near-singular fits, NaNs
    sneaking through an intermediate step, etc.).
    """
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    fracs_arr = np.asarray(fracs, dtype=np.float64)
    L = ca0.shape[0]
    Ti = len(fracs_arr)

    def _linear_all():
        out = np.empty((Ti, L, 3), dtype=np.float64)
        for i, s in enumerate(fracs_arr):
            sc = float(np.clip(s, 0.0, 1.0))
            out[i] = (1.0 - sc) * ca0 + sc * ca1
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        out = _predict_interior_impl(ca0, ca1, fracs_arr, mask)
        out = np.asarray(out, dtype=np.float64)
        if out.shape != (Ti, L, 3) or not np.all(np.isfinite(out)):
            return _linear_all()
        return out
    except Exception:
        return _linear_all()


# EVOLVE-BLOCK-END


def run_interpolation(ca_start, ca_end, fracs, mask):
    """Fixed entry point used by the evaluator."""
    return predict_interior(ca_start, ca_end, fracs, mask)