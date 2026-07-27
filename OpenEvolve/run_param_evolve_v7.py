#!/usr/bin/env python
"""v7: Creative selection for path-interpolator param evolution (local GPU, no API).

Upgrades over v6 exploit-heavy loop:
  - Archive (top-K by dual_min) + multi-objective Pareto niches
  - 4 islands with different mutation / sampling biases
  - Novelty archive (L2 in normalized param space)
  - Crossover + creative mutation operators (large jump, block reset, structural flip, mirror)
  - Adaptive explore / exploit / cross / wild schedule
  - Structural algo_mode discrete modes (not only continuous weights)
  - Immigrant injection from candidates/*.py PARAMS every 100 iters

Fitness contract unchanged: dual_min = min(evolve, holdout); fold-weighted combined_score.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
EX = ROOT / "examples" / "human_interp_path"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EX))
os.environ.setdefault("HUMAN_EVOLVE_DEVICE", "cuda")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import importlib.util

spec_e = importlib.util.spec_from_file_location("evaluator", EX / "evaluator.py")
ev = importlib.util.module_from_spec(spec_e)
spec_e.loader.exec_module(ev)

OUT = EX / "openevolve_output_grok"
TRIAL = EX / "_param_trial_v7.py"
CAND_DIR = EX / "candidates"
BEST_DIR = OUT / "best"

# algo_mode: 0 screw_local, 1 multiscale_vote, 2 flex_half,
#            3 bond_heavy, 4 geodesic_lite, 5 ensemble_local
ALGO_MODES = {
    0: "screw_local",
    1: "multiscale_vote",
    2: "flex_half",
    3: "bond_heavy",
    4: "geodesic_lite",
    5: "ensemble_local",
}

# Keys used for novelty / distance (continuous + scaled ints)
NOVELTY_KEYS = [
    "ease", "hermite", "w_lin", "w_screw", "w_local", "w_dom", "w_mode",
    "local_half", "half_s", "half_l", "w_ms_small", "w_ms_mid", "w_ms_large",
    "n_dom", "min_dom", "use_chasles", "timewarp", "s_gate", "n_modes",
    "mode_mid_boost", "bond_strength", "algo_mode",
]

PROG = r'''
import numpy as np
PARAMS = {}

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
'''


def materialize(params: dict, path: Path) -> None:
    src = PROG.replace("PARAMS = {}", "PARAMS = " + repr(params), 1)
    path.write_text(src, encoding="utf-8")


def balanced(es: float, hs: float) -> float:
    return 0.5 * es + 0.5 * hs - 0.35 * max(0.0, -hs)


def clip(x, a, b):
    return a if x < a else b if x > b else x


def ensure_params(p: dict) -> dict:
    """Fill defaults + repair constraints."""
    q = copy.deepcopy(p)
    defaults = dict(
        ease=0.0, hermite=0.0, w_lin=0.1, w_screw=0.5, w_local=0.3, w_dom=0.0, w_mode=0.1,
        local_half=20, half_s=3, half_l=26, w_ms_small=0.2, w_ms_mid=0.75, w_ms_large=0.05,
        n_dom=1, min_dom=8, use_chasles=0.0, timewarp=0.05, s_gate=0.15, n_modes=1,
        mode_mid_boost=0.2, bond_strength=0.0, algo_mode=0,
    )
    for k, v in defaults.items():
        if k not in q:
            q[k] = v
    q["algo_mode"] = int(clip(int(q.get("algo_mode", 0)), 0, 5))
    q["local_half"] = int(clip(int(q["local_half"]), 8, 36))
    q["half_s"] = int(clip(int(q["half_s"]), 2, 12))
    q["half_l"] = int(clip(int(q["half_l"]), 14, 40))
    q["n_dom"] = int(clip(int(q["n_dom"]), 1, 6))
    q["min_dom"] = int(clip(int(q["min_dom"]), 4, 16))
    q["n_modes"] = int(clip(int(q["n_modes"]), 0, 3))
    for k in ("ease", "hermite", "use_chasles", "timewarp", "s_gate",
              "mode_mid_boost", "bond_strength"):
        q[k] = float(clip(float(q[k]), 0.0, 1.0))
    ws = ["w_lin", "w_screw", "w_local", "w_dom", "w_mode"]
    s = sum(max(0.0, float(q[k])) for k in ws) + 1e-12
    for k in ws:
        q[k] = max(0.0, float(q[k])) / s
    ms = ["w_ms_small", "w_ms_mid", "w_ms_large"]
    s2 = sum(max(0.0, float(q[k])) for k in ms) + 1e-12
    for k in ms:
        q[k] = max(0.0, float(q[k])) / s2
    if q["half_l"] <= q["local_half"]:
        q["half_l"] = int(q["local_half"]) + 4
    if q["half_s"] >= q["local_half"]:
        q["half_s"] = max(2, int(q["local_half"]) - 3)
    return q


def param_vec(p: dict) -> List[float]:
    """Normalized feature vector for novelty."""
    p = ensure_params(p)
    scales = {
        "local_half": 36.0, "half_s": 12.0, "half_l": 40.0,
        "n_dom": 6.0, "min_dom": 16.0, "n_modes": 3.0, "algo_mode": 5.0,
    }
    v = []
    for k in NOVELTY_KEYS:
        x = float(p.get(k, 0.0))
        v.append(x / scales.get(k, 1.0))
    return v


def l2(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def sample_wild(rng: random.Random, island: int = 3) -> dict:
    """Unconstrained / anti-bias sample (creative exploration)."""
    raw = [rng.random() for _ in range(5)]
    if island == 0:
        # high-local / multiscale
        raw[0] *= 0.25
        raw[2] = raw[2] * 0.5 + 0.5
        raw[3] *= 0.2
    elif island == 1:
        # high-screw / low-local
        raw[1] = raw[1] * 0.4 + 0.6
        raw[2] *= 0.35
    elif island == 2:
        # structural modes
        pass
    else:
        # fully wild
        pass
    s = sum(raw) + 1e-12
    w = [x / s for x in raw]
    if island == 0 and rng.random() > 0.2:
        small = rng.uniform(0.15, 0.55)
        large = rng.uniform(0.0, 0.08)
        mid = max(0.05, 1.0 - small - large)
        ms = [small, mid, large]
    elif island == 1:
        ms = [rng.uniform(0.0, 0.15), rng.uniform(0.7, 1.0), rng.uniform(0.0, 0.1)]
    else:
        ms = [rng.random(), rng.random(), rng.random()]
    sms = sum(ms) + 1e-12
    ms = [x / sms for x in ms]
    mode = 0
    if island == 2:
        mode = rng.choice([1, 2, 3, 4, 5, 2, 3])
    elif island == 3:
        mode = rng.choice([0, 0, 1, 2, 3, 4, 5])
    elif rng.random() < 0.12:
        mode = rng.randint(0, 5)
    return ensure_params({
        "ease": rng.uniform(0.0, 0.25),
        "hermite": rng.choice([0.0, 0.0, 0.0, 0.05, 0.1, 0.15]),
        "w_lin": w[0], "w_screw": w[1], "w_local": w[2], "w_dom": w[3], "w_mode": w[4],
        "local_half": rng.randint(12, 32),
        "half_s": rng.choice([2, 2, 3, 4, 5]),
        "half_l": rng.choice([20, 24, 28, 32, 36]),
        "w_ms_small": ms[0], "w_ms_mid": ms[1], "w_ms_large": ms[2],
        "n_dom": rng.choice([1, 1, 2, 3, 4]),
        "min_dom": rng.choice([6, 8, 10]),
        "use_chasles": rng.choice([0.0, 0.0, 0.05, 0.1, 0.2]),
        "timewarp": rng.uniform(0.0, 0.2),
        "s_gate": rng.uniform(0.0, 0.35),
        "n_modes": rng.choice([0, 1, 1, 1, 2]),
        "mode_mid_boost": rng.uniform(0.0, 0.35),
        "bond_strength": rng.choice([0.0, 0.0, 0.02, 0.05, 0.1, 0.15]),
        "algo_mode": mode,
    })


def sample_biased(rng: random.Random) -> dict:
    """v6-style high-local bias sample."""
    return sample_wild(rng, island=0)


def repair(q: dict) -> dict:
    return ensure_params(q)


def mutate_soft(p: dict, rng: random.Random, scale: float = 0.35) -> dict:
    q = copy.deepcopy(p)
    keys = [k for k in q.keys() if k != "algo_mode"]
    for k in rng.sample(keys, k=min(len(keys), rng.randint(1, 5))):
        if k in ("local_half", "half_s", "half_l", "n_dom", "min_dom", "n_modes"):
            step = rng.choice([-2, -1, 1, 2])
            q[k] = int(q[k]) + step
        elif k.startswith("w_"):
            q[k] = float(max(0.0, float(q[k]) + rng.gauss(0, 0.08 * scale)))
        else:
            q[k] = float(float(q[k]) + rng.gauss(0, 0.08 * scale))
    return repair(q)


def mutate_large_jump(p: dict, rng: random.Random) -> dict:
    return mutate_soft(p, rng, scale=1.1)


def mutate_block_reset(p: dict, rng: random.Random) -> dict:
    q = copy.deepcopy(p)
    block = rng.choice(["blend", "multiscale", "time", "struct", "domain"])
    if block == "blend":
        raw = [rng.random() for _ in range(5)]
        s = sum(raw) + 1e-12
        for k, v in zip(["w_lin", "w_screw", "w_local", "w_dom", "w_mode"], raw):
            q[k] = v / s
    elif block == "multiscale":
        ms = [rng.random(), rng.random(), rng.random()]
        s = sum(ms) + 1e-12
        q["w_ms_small"], q["w_ms_mid"], q["w_ms_large"] = ms[0] / s, ms[1] / s, ms[2] / s
        q["half_s"] = rng.choice([2, 3, 4])
        q["half_l"] = rng.choice([22, 26, 30, 36])
        q["local_half"] = rng.randint(16, 30)
    elif block == "time":
        q["ease"] = rng.uniform(0.0, 0.2)
        q["hermite"] = rng.choice([0.0, 0.05, 0.1])
        q["timewarp"] = rng.uniform(0.0, 0.18)
        q["s_gate"] = rng.uniform(0.0, 0.3)
    elif block == "struct":
        q["algo_mode"] = rng.randint(0, 5)
        q["bond_strength"] = rng.choice([0.0, 0.05, 0.1, 0.15])
        q["use_chasles"] = rng.choice([0.0, 0.05, 0.15])
    else:
        q["n_dom"] = rng.choice([1, 2, 3, 4])
        q["min_dom"] = rng.choice([6, 8, 10])
        q["n_modes"] = rng.choice([0, 1, 2])
        q["mode_mid_boost"] = rng.uniform(0.05, 0.35)
    return repair(q)


def mutate_structural_flip(p: dict, rng: random.Random) -> dict:
    q = copy.deepcopy(p)
    q["algo_mode"] = (int(q.get("algo_mode", 0)) + rng.choice([1, 2, 3, 4, 5])) % 6
    if rng.random() < 0.5:
        q["n_dom"] = rng.choice([1, 2, 3, 4, 5])
    if rng.random() < 0.4:
        q["use_chasles"] = 0.0 if float(q.get("use_chasles", 0)) > 0.05 else rng.uniform(0.1, 0.4)
    return repair(q)


def mutate_mirror_elite(p: dict, rng: random.Random) -> dict:
    """Invert residual bias: swap high local <-> high screw."""
    q = copy.deepcopy(p)
    q["w_local"], q["w_screw"] = float(q.get("w_screw", 0.4)), float(q.get("w_local", 0.4))
    # flip multiscale emphasis
    q["w_ms_small"], q["w_ms_mid"] = float(q.get("w_ms_mid", 0.5)), float(q.get("w_ms_small", 0.3))
    if rng.random() < 0.5:
        q["s_gate"] = float(clip(1.0 - float(q.get("s_gate", 0.15)), 0.0, 0.4))
    return repair(q)


def mutate(p: dict, rng: random.Random, island: int = 0) -> dict:
    """Creative multi-operator mutation."""
    r = rng.random()
    # island 2 prefers structural ops
    if island == 2:
        weights = (0.30, 0.15, 0.20, 0.25, 0.10)  # soft, jump, block, struct, mirror
    elif island == 3:
        weights = (0.25, 0.30, 0.20, 0.15, 0.10)
    else:
        weights = (0.50, 0.20, 0.15, 0.10, 0.05)
    c0, c1, c2, c3 = weights[0], sum(weights[:2]), sum(weights[:3]), sum(weights[:4])
    if r < c0:
        scale = 0.18 + 0.4 * rng.random()
        if island == 1:
            scale *= 0.85
        return mutate_soft(p, rng, scale=scale)
    if r < c1:
        return mutate_large_jump(p, rng)
    if r < c2:
        return mutate_block_reset(p, rng)
    if r < c3:
        return mutate_structural_flip(p, rng)
    return mutate_mirror_elite(p, rng)


def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    """Continuous blend + integer gene pick + occasional block gene-swap."""
    a, b = ensure_params(a), ensure_params(b)
    # Beta(2,2) ~ peaked at 0.5
    alpha = rng.betavariate(2.0, 2.0)
    q: Dict[str, Any] = {}
    cont = [
        "ease", "hermite", "w_lin", "w_screw", "w_local", "w_dom", "w_mode",
        "w_ms_small", "w_ms_mid", "w_ms_large", "use_chasles", "timewarp",
        "s_gate", "mode_mid_boost", "bond_strength",
    ]
    ints = ["local_half", "half_s", "half_l", "n_dom", "min_dom", "n_modes", "algo_mode"]
    for k in cont:
        q[k] = alpha * float(a[k]) + (1.0 - alpha) * float(b[k])
    for k in ints:
        q[k] = a[k] if rng.random() < 0.5 else b[k]
        if rng.random() < 0.35 and k != "algo_mode":
            q[k] = int(q[k]) + rng.choice([-1, 1])
    # 20% gene-swap blocks
    if rng.random() < 0.20:
        if rng.random() < 0.5:
            for k in ("w_ms_small", "w_ms_mid", "w_ms_large", "half_s", "half_l", "local_half"):
                q[k] = a[k]
            for k in ("w_screw", "w_local", "use_chasles", "timewarp"):
                q[k] = b[k]
        else:
            for k in ("w_ms_small", "w_ms_mid", "w_ms_large", "half_s", "half_l", "local_half"):
                q[k] = b[k]
            for k in ("w_screw", "w_local", "use_chasles", "timewarp"):
                q[k] = a[k]
        q["algo_mode"] = a["algo_mode"] if rng.random() < 0.5 else b["algo_mode"]
    return repair(q)


def dominates(a: dict, b: dict) -> bool:
    """Pareto on dual_min, fold proxy (use evolve as fold-ish), balanced, -COM gap.
    Higher dual_min, higher balanced, higher evolve better; lower |e-h| better as soft 4th.
    """
    da, db = a["dual_min"], b["dual_min"]
    ba, bb = a["balanced"], b["balanced"]
    ea, eb = a["evolve"], b["evolve"]
    ga = -abs(a["evolve"] - a["holdout"])
    gb = -abs(b["evolve"] - b["holdout"])
    better_or_eq = da >= db and ba >= bb and ea >= eb and ga >= gb
    strictly = da > db or ba > bb or ea > eb or ga > gb
    return better_or_eq and strictly


# ---------------------------------------------------------------------------
# Population structures
# ---------------------------------------------------------------------------

class CreativePopulation:
    def __init__(self, n_islands: int = 4, archive_k: int = 20, novelty_k: int = 30):
        self.n_islands = n_islands
        self.archive_k = archive_k
        self.novelty_k = novelty_k
        self.islands: List[List[dict]] = [[] for _ in range(n_islands)]
        self.archive: List[dict] = []  # by dual_min
        self.pareto: List[dict] = []
        self.novelty: List[dict] = []  # diverse params
        self.best_dual: Optional[dict] = None
        self.best_bal: Optional[dict] = None
        self.best_fold: Optional[dict] = None  # highest evolve (fold-weighted combined)
        self.sel_counts = {"explore": 0, "exploit": 0, "cross": 0, "wild": 0, "immigrant": 0}

    def _push_sorted(self, lst: List[dict], rec: dict, key: str, k: int, reverse=True):
        lst.append(rec)
        lst.sort(key=lambda r: r.get(key, -9), reverse=reverse)
        del lst[k:]

    def add(self, rec: dict, island: int = 0):
        island = int(island) % self.n_islands
        self.islands[island].append(rec)
        # cap island size
        if len(self.islands[island]) > 40:
            self.islands[island].sort(key=lambda r: r.get("dual_min", -9), reverse=True)
            self.islands[island] = self.islands[island][:25]

        dscore = rec.get("dual_score", rec.get("dual_min", -9))
        if self.best_dual is None or dscore > self.best_dual.get("dual_score", -9) + 1e-9:
            self.best_dual = rec
        if self.best_bal is None or rec["balanced"] > self.best_bal["balanced"] + 1e-9:
            self.best_bal = rec
        if self.best_fold is None or rec["evolve"] > self.best_fold["evolve"] + 1e-9:
            self.best_fold = rec

        self._push_sorted(self.archive, rec, "dual_min", self.archive_k)

        # Pareto update
        self.pareto.append(rec)
        nondom = []
        for cand in self.pareto:
            if any(dominates(other, cand) for other in self.pareto if other is not cand):
                continue
            nondom.append(cand)
        # unique by dual_min roughly
        seen = set()
        uniq = []
        for c in nondom:
            sig = (round(c["dual_min"], 6), round(c["balanced"], 6), int(c["params"].get("algo_mode", 0)))
            if sig in seen:
                continue
            seen.add(sig)
            uniq.append(c)
        uniq.sort(key=lambda r: r["dual_min"], reverse=True)
        self.pareto = uniq[:30]

        # novelty archive: keep if far from existing
        pv = param_vec(rec["params"])
        if not self.novelty:
            self.novelty.append(rec)
        else:
            mind = min(l2(pv, param_vec(x["params"])) for x in self.novelty)
            if mind > 0.35 or rec["dual_min"] > (self.archive[0]["dual_min"] if self.archive else -1) - 0.002:
                self.novelty.append(rec)
                # if too many, drop lowest dual among closest pairs later
                if len(self.novelty) > self.novelty_k:
                    self.novelty.sort(key=lambda r: r.get("dual_min", -9), reverse=True)
                    # keep top half dual + most novel of rest
                    keep = self.novelty[: self.novelty_k // 2]
                    rest = self.novelty[self.novelty_k // 2 :]
                    # greedy farthest
                    while len(keep) < self.novelty_k and rest:
                        best_i, best_d = 0, -1.0
                        for i, r in enumerate(rest):
                            d = min(l2(param_vec(r["params"]), param_vec(k["params"])) for k in keep)
                            if d > best_d:
                                best_d, best_i = d, i
                        keep.append(rest.pop(best_i))
                    self.novelty = keep

    def island_best(self, island: int) -> Optional[dict]:
        pop = self.islands[island]
        if not pop:
            return None
        return max(pop, key=lambda r: r.get("dual_min", -9))

    def tournament(self, pool: List[dict], rng: random.Random, k: int = 3) -> Optional[dict]:
        if not pool:
            return None
        cand = [rng.choice(pool) for _ in range(min(k, len(pool)))]
        return max(cand, key=lambda r: r.get("dual_score", r.get("dual_min", -9)))

    def migrate(self, rng: random.Random, n: int = 2):
        elites = []
        for i in range(self.n_islands):
            b = self.island_best(i)
            if b is not None:
                elites.append((i, b))
        if len(elites) < 2:
            return
        for _ in range(n):
            src_i, src = rng.choice(elites)
            dst = (src_i + rng.randint(1, self.n_islands - 1)) % self.n_islands
            self.islands[dst].append(copy.deepcopy(src))

    def diversity_stats(self) -> dict:
        modes = {}
        for r in self.archive:
            m = int(r["params"].get("algo_mode", 0))
            modes[ALGO_MODES.get(m, str(m))] = modes.get(ALGO_MODES.get(m, str(m)), 0) + 1
        # recipe clusters by coarse weight signature
        sigs = set()
        for r in self.archive:
            p = r["params"]
            sig = (
                round(float(p.get("w_local", 0)), 1),
                round(float(p.get("w_screw", 0)), 1),
                round(float(p.get("w_ms_small", 0)), 1),
                int(p.get("local_half", 0) // 4),
                int(p.get("algo_mode", 0)),
            )
            sigs.add(sig)
        return {
            "archive_size": len(self.archive),
            "pareto_size": len(self.pareto),
            "novelty_size": len(self.novelty),
            "modes_in_archive": modes,
            "distinct_recipes": len(sigs),
            "sel_counts": dict(self.sel_counts),
        }


def schedule_ratios(it: int, n_iters: int) -> Tuple[float, float, float, float]:
    """Return explore, exploit, cross, wild ratios (sum=1)."""
    if it < min(100, n_iters // 5):
        return 0.45, 0.25, 0.15, 0.15
    if it > max(n_iters - 150, int(0.7 * n_iters)):
        return 0.25, 0.45, 0.15, 0.15
    return 0.35, 0.35, 0.15, 0.15


def select_parent(
    pop: CreativePopulation, rng: random.Random, it: int, n_iters: int, island: int
) -> Tuple[dict, str]:
    """Creative parent selection. Returns (params, selection_kind)."""
    explore, exploit, cross, wild = schedule_ratios(it, n_iters)
    r = rng.random()
    # Island 2/3 push more explore
    if island >= 2:
        explore = min(0.55, explore + 0.1)
        exploit = max(0.15, exploit - 0.1)

    def any_params():
        if pop.archive:
            return copy.deepcopy(rng.choice(pop.archive)["params"])
        if pop.best_dual:
            return copy.deepcopy(pop.best_dual["params"])
        return sample_wild(rng, island)

    if r < explore:
        pop.sel_counts["explore"] += 1
        pool_choice = rng.random()
        if pool_choice < 0.40 and pop.islands[island]:
            rec = rng.choice(pop.islands[island])
            return copy.deepcopy(rec["params"]), "explore_island"
        if pool_choice < 0.70 and pop.novelty:
            rec = rng.choice(pop.novelty)
            return copy.deepcopy(rec["params"]), "explore_novelty"
        if pool_choice < 0.90 and pop.pareto:
            rec = rng.choice(pop.pareto)
            return copy.deepcopy(rec["params"]), "explore_pareto"
        return sample_wild(rng, island), "explore_sample"

    if r < explore + exploit:
        pop.sel_counts["exploit"] += 1
        if rng.random() < 0.6 and pop.archive:
            rec = pop.tournament(pop.archive, rng, k=3)
            if rec:
                return copy.deepcopy(rec["params"]), "exploit_archive"
        ib = pop.island_best(island)
        if ib is not None:
            return copy.deepcopy(ib["params"]), "exploit_island"
        if pop.best_dual is not None:
            return copy.deepcopy(pop.best_dual["params"]), "exploit_dual"
        return any_params(), "exploit_fallback"

    if r < explore + exploit + cross:
        pop.sel_counts["cross"] += 1
        pool = pop.archive if len(pop.archive) >= 2 else (
            [x for isl in pop.islands for x in isl] if any(pop.islands) else []
        )
        if len(pool) >= 2:
            a = pop.tournament(pool, rng, k=3)
            b = pop.tournament(pool, rng, k=3)
            if a is not None and b is not None:
                return crossover(a["params"], b["params"], rng), "cross"
        return mutate(any_params(), rng, island), "cross_fallback"

    pop.sel_counts["wild"] += 1
    return sample_wild(rng, island), "wild"


# v6 dual + balanced + neighborhood + mode probes
def build_seeds() -> List[dict]:
    v6_dual = dict(
        ease=0.14197890234079205, hermite=0.11370771013186731,
        w_lin=0.05471703032739539, w_screw=0.3889218955523097,
        w_local=0.4489436579535165, w_dom=0.04320915593297041,
        w_mode=0.06420826023280803, local_half=28, half_s=2, half_l=36,
        w_ms_small=0.5809727782284834, w_ms_mid=0.41902722177051643, w_ms_large=0.0,
        n_dom=1, min_dom=8, use_chasles=0.034091927148485454, timewarp=0.0,
        s_gate=0.001999113056379582, n_modes=1, mode_mid_boost=0.2895873764860074,
        bond_strength=0.0, algo_mode=0,
    )
    v6_bal = dict(
        ease=0.14324942113249264, hermite=0.059022665775652905,
        w_lin=0.03369241154320932, w_screw=0.49998862472710665,
        w_local=0.4147807875459672, w_dom=0.0, w_mode=0.05153817618270478,
        local_half=25, half_s=2, half_l=36, w_ms_small=0.5299984862219004,
        w_ms_mid=0.4280172482711877, w_ms_large=0.04198426550591202,
        n_dom=1, min_dom=10, use_chasles=0.006370318953902556, timewarp=0.00629149604588762,
        s_gate=0.0985531787266622, n_modes=1, mode_mid_boost=0.2679192325718954,
        bond_strength=0.0, algo_mode=0,
    )
    linear = dict(
        ease=0, hermite=0, w_lin=1, w_screw=0, w_local=0, w_dom=0, w_mode=0,
        local_half=13, half_s=5, half_l=22, w_ms_small=0.0, w_ms_mid=1.0, w_ms_large=0.0,
        n_dom=1, min_dom=8, use_chasles=0, timewarp=0, s_gate=0, n_modes=0,
        mode_mid_boost=0, bond_strength=0, algo_mode=0,
    )
    seeds = [linear, v6_dual, v6_bal]
    # structural mode probes from v6 dual
    for mode in (1, 2, 3, 4, 5):
        p = copy.deepcopy(v6_dual)
        p["algo_mode"] = mode
        if mode == 3:
            p["bond_strength"] = 0.12
        if mode == 4:
            p["w_local"] = 0.55
        seeds.append(p)
    # neighborhood probes
    probes = [
        dict(ease=0.0, hermite=0.0, w_lin=0.0, w_screw=0.45, w_local=0.45, w_dom=0.0, w_mode=0.10,
             local_half=26, half_s=2, half_l=32, w_ms_small=0.45, w_ms_mid=0.50, w_ms_large=0.05,
             n_dom=1, min_dom=8, use_chasles=0.0, timewarp=0.05, s_gate=0.15, n_modes=1,
             mode_mid_boost=0.28, bond_strength=0.0, algo_mode=0),
        dict(ease=0.05, hermite=0.05, w_lin=0.02, w_screw=0.55, w_local=0.35, w_dom=0.0, w_mode=0.08,
             local_half=22, half_s=3, half_l=28, w_ms_small=0.30, w_ms_mid=0.65, w_ms_large=0.05,
             n_dom=2, min_dom=8, use_chasles=0.1, timewarp=0.08, s_gate=0.20, n_modes=1,
             mode_mid_boost=0.22, bond_strength=0.05, algo_mode=1),
        dict(ease=0.0, hermite=0.0, w_lin=0.05, w_screw=0.30, w_local=0.55, w_dom=0.05, w_mode=0.05,
             local_half=24, half_s=2, half_l=34, w_ms_small=0.4, w_ms_mid=0.5, w_ms_large=0.1,
             n_dom=1, min_dom=8, use_chasles=0.0, timewarp=0.0, s_gate=0.05, n_modes=1,
             mode_mid_boost=0.30, bond_strength=0.08, algo_mode=2),
    ]
    seeds.extend(probes)
    return [ensure_params(s) for s in seeds]


_PARAMS_RE = re.compile(r"PARAMS\s*=\s*(\{.*?\})", re.DOTALL)


def extract_params_from_py(path: Path) -> Optional[dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    m = _PARAMS_RE.search(text)
    if not m:
        return None
    try:
        # safe-ish: only dict literal from our own candidates
        p = eval(m.group(1), {"__builtins__": {}}, {})
        if isinstance(p, dict) and ("w_local" in p or "w_screw" in p or "w_lin" in p):
            return ensure_params(p)
    except Exception:
        return None
    return None


def load_candidate_immigrants(rng: random.Random, max_n: int = 12) -> List[Tuple[str, dict]]:
    """Load param-compatible candidates + best programs as immigrants."""
    found: List[Tuple[str, dict]] = []
    paths: List[Path] = []
    if CAND_DIR.is_dir():
        paths.extend(sorted(CAND_DIR.glob("*.py")))
    if BEST_DIR.is_dir():
        paths.extend(sorted(BEST_DIR.glob("best_program*.py")))
    for path in paths:
        if path.name.startswith("_") or path.name.startswith("gen_"):
            continue
        p = extract_params_from_py(path)
        if p is not None:
            found.append((path.name, p))
    if not found:
        return []
    # mix top alphabetical (often numbered champs) + random
    rng.shuffle(found)
    return found[:max_n]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="OpenEvolve v7 creative local param evolution")
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--screen-windows", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--islands", type=int, default=4)
    ap.add_argument("--archive-k", type=int, default=20)
    ap.add_argument("--migrate-every", type=int, default=50)
    ap.add_argument("--immigrant-every", type=int, default=100)
    args = ap.parse_args()
    os.environ["HUMAN_EVOLVE_DEVICE"] = args.device

    OUT.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "param_evolve_v7_log.jsonl"
    rng = random.Random(args.seed)
    pop = CreativePopulation(n_islands=args.islands, archive_k=args.archive_k)
    t0 = time.perf_counter()
    immigrants_cache = load_candidate_immigrants(rng, max_n=16)

    def eval_file_score(path, n_windows, split):
        return ev._score(str(path), n_windows=n_windows, split=split, timeout=180)

    def consider(params, it, force_full=False, screen_sc=None, island=0, sel_kind="seed"):
        params = ensure_params(params)
        materialize(params, TRIAL)
        if screen_sc is None:
            scr = eval_file_score(TRIAL, args.screen_windows, "evolve")
            screen_sc = float(scr.get("combined_score", -1))
            screen_rmsd = float(scr.get("model_rmsd", 99))
            screen_fold = float(scr.get("folding_score", scr.get("combined_score", -1)))
        else:
            screen_rmsd = None
            screen_fold = None
        thr = -0.006
        if pop.best_dual is not None:
            thr = min(thr, pop.best_dual.get("screen", -9) - 0.01)
        if pop.best_bal is not None:
            thr = min(thr, pop.best_bal.get("screen", -9) - 0.004)
        # explore structural modes a bit more generously on first screens
        if int(params.get("algo_mode", 0)) != 0 and screen_sc > thr - 0.004:
            thr = screen_sc - 1e-9  # allow full if not terrible
        do_full = force_full or pop.best_dual is None or pop.best_bal is None or screen_sc > thr
        if not do_full:
            return screen_sc, None, screen_rmsd
        full_e = eval_file_score(TRIAL, None, "evolve")
        full_h = eval_file_score(TRIAL, None, "holdout")
        es = float(full_e.get("combined_score", -1))
        hs = float(full_h.get("combined_score", -1))
        bal = balanced(es, hs)
        dual = min(es, hs)
        dscore = dual
        if es > 0 and hs > 0:
            gap = abs(es - hs)
            dscore = dual + 0.12 * dual / (1.0 + 25.0 * gap)
        rec = {
            "params": copy.deepcopy(params),
            "screen": screen_sc,
            "evolve": es,
            "holdout": hs,
            "balanced": bal,
            "dual_min": dual,
            "dual_score": dscore,
            "metrics_e": full_e,
            "metrics_h": full_h,
            "iter": it,
            "island": island,
            "sel_kind": sel_kind,
            "algo_mode": int(params.get("algo_mode", 0)),
            "algo_name": ALGO_MODES.get(int(params.get("algo_mode", 0)), "?"),
        }
        prev_dscore = pop.best_dual.get("dual_score", -9) if pop.best_dual else -9.0
        prev_bal = pop.best_bal["balanced"] if pop.best_bal else -9.0
        is_new_dual = pop.best_dual is None or dscore > prev_dscore + 1e-9
        is_new_bal = pop.best_bal is None or bal > prev_bal + 1e-9
        pop.add(rec, island=island)
        if is_new_bal:
            print(
                f"[{it:4d}] NEW BAL   bal={bal:+.5f}  e={es:+.5f} ({full_e.get('model_rmsd'):.4f}A)  "
                f"h={hs:+.5f}  dual={dual:+.5f}  mode={rec['algo_name']}  via={sel_kind}",
                flush=True,
            )
        if is_new_dual:
            print(
                f"[{it:4d}] NEW DUAL  dscore={dscore:+.5f} min={dual:+.5f}  e={es:+.5f} h={hs:+.5f}  "
                f"mode={rec['algo_name']} half={params.get('local_half')} "
                f"loc={params.get('w_local'):.2f} ms={params.get('w_ms_small'):.2f}  via={sel_kind}",
                flush=True,
            )
        return screen_sc, rec, float(full_e.get("model_rmsd", 99))

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print(
        f"=== param evolve v7 CREATIVE: {args.iterations} iters device={args.device} "
        f"islands={args.islands} archive_k={args.archive_k} immigrants={len(immigrants_cache)} ===",
        flush=True,
    )
    print(f"  modes: {ALGO_MODES}", flush=True)

    with open(log_path, "w", encoding="utf-8") as logf:
        seeds = build_seeds()
        for i, seed in enumerate(seeds):
            isl = i % args.islands
            sc, rec, _ = consider(seed, it=-(i + 1), force_full=True, island=isl, sel_kind="seed")
            row = {"iter": -(i + 1), "seed": True, "screen": sc, "island": isl,
                   "algo_mode": seed.get("algo_mode", 0)}
            if rec:
                row.update({k: rec[k] for k in (
                    "evolve", "holdout", "balanced", "dual_min", "params", "algo_name", "sel_kind"
                )})
            logf.write(json.dumps(row) + "\n")
            logf.flush()

        # inject a few immigrants at start (screen then full if competitive)
        for j, (name, ip) in enumerate(immigrants_cache[:6]):
            isl = j % args.islands
            pop.sel_counts["immigrant"] += 1
            sc, rec, _ = consider(ip, it=-(100 + j), force_full=False, island=isl, sel_kind=f"immigrant:{name}")
            row = {"iter": -(100 + j), "immigrant": name, "screen": sc, "island": isl}
            if rec:
                row.update({k: rec[k] for k in ("evolve", "holdout", "balanced", "dual_min", "params")})
            logf.write(json.dumps(row) + "\n")
            logf.flush()

        for it in range(args.iterations):
            island = it % args.islands
            parent, sel_kind = select_parent(pop, rng, it, args.iterations, island)
            # After selection, apply mutation unless pure wild sample or pure crossover product
            if sel_kind in ("wild", "explore_sample", "cross"):
                params = parent if sel_kind != "wild" else parent
                if sel_kind == "wild":
                    params = parent
                elif sel_kind == "cross":
                    # light polish after crossover
                    if rng.random() < 0.5:
                        params = mutate_soft(parent, rng, scale=0.15)
                    else:
                        params = parent
                else:
                    params = parent
            else:
                params = mutate(parent, rng, island=island)

            # periodic immigrants
            if args.immigrant_every > 0 and (it + 1) % args.immigrant_every == 0 and immigrants_cache:
                pop.sel_counts["immigrant"] += 1
                name, ip = rng.choice(immigrants_cache)
                params = mutate(ip, rng, island=island) if rng.random() < 0.7 else ip
                sel_kind = f"immigrant:{name}"

            sc, rec, srmsd = consider(params, it=it, island=island, sel_kind=sel_kind)
            row = {
                "iter": it, "screen": sc, "screen_rmsd": srmsd, "params": params,
                "island": island, "sel_kind": sel_kind,
                "algo_mode": int(params.get("algo_mode", 0)),
            }
            if rec:
                row.update({
                    "evolve": rec["evolve"], "holdout": rec["holdout"],
                    "balanced": rec["balanced"], "dual_min": rec["dual_min"],
                    "dual_score": rec.get("dual_score"), "algo_name": rec.get("algo_name"),
                })
            else:
                row.update({
                    "evolve": None, "holdout": None, "balanced": None,
                    "dual_min": None, "dual_score": None,
                })
            logf.write(json.dumps(row) + "\n")
            logf.flush()

            if args.migrate_every > 0 and (it + 1) % args.migrate_every == 0:
                pop.migrate(rng, n=2)

            if (it + 1) % 50 == 0:
                stats = pop.diversity_stats()
                bd = pop.best_dual
                bb = pop.best_bal
                print(
                    f"  ... {it+1}/{args.iterations}  "
                    f"dual_min={bd['dual_min'] if bd else None}  "
                    f"bal={bb['balanced'] if bb else None}  "
                    f"archive={stats['archive_size']} recipes={stats['distinct_recipes']}  "
                    f"modes={stats['modes_in_archive']}  "
                    f"sel={stats['sel_counts']}  "
                    f"elapsed={time.perf_counter()-t0:.0f}s",
                    flush=True,
                )
                ck = {
                    "iter": it,
                    "best_dual": None if bd is None else {
                        "iter": bd["iter"], "dual_min": bd["dual_min"],
                        "dual_score": bd.get("dual_score"), "evolve": bd["evolve"],
                        "holdout": bd["holdout"], "balanced": bd["balanced"],
                        "params": bd["params"], "algo_name": bd.get("algo_name"),
                    },
                    "best_balanced": None if bb is None else {
                        "iter": bb["iter"], "balanced": bb["balanced"],
                        "evolve": bb["evolve"], "holdout": bb["holdout"],
                        "dual_min": bb["dual_min"], "params": bb["params"],
                    },
                    "diversity": stats,
                    "elapsed_s": time.perf_counter() - t0,
                }
                (OUT / "param_evolve_v7_checkpoint.json").write_text(
                    json.dumps(ck, indent=2), encoding="utf-8"
                )
                if bd is not None:
                    materialize(bd["params"], BEST_DIR / "best_program_v7_dual_ckpt.py")
                if bb is not None:
                    materialize(bb["params"], BEST_DIR / "best_program_v7_bal_ckpt.py")

    promo = pop.best_dual if pop.best_dual is not None else pop.best_bal
    if promo is not None:
        materialize(promo["params"], BEST_DIR / "best_program_v7.py")
        materialize(promo["params"], BEST_DIR / "best_program.py")
        materialize(promo["params"], CAND_DIR / "34_v7_dual_min.py")
        if pop.best_bal is not None:
            materialize(pop.best_bal["params"], BEST_DIR / "best_program_v7_balanced.py")
            materialize(pop.best_bal["params"], CAND_DIR / "35_v7_balanced.py")
        stats = pop.diversity_stats()
        info = {
            "source": "param_evolve_v7_creative",
            "iterations": args.iterations,
            "device": args.device,
            "islands": args.islands,
            "best_dual": None if pop.best_dual is None else {
                "iter": pop.best_dual["iter"],
                "dual_min": pop.best_dual["dual_min"],
                "dual_score": pop.best_dual.get("dual_score"),
                "evolve": pop.best_dual["evolve"],
                "holdout": pop.best_dual["holdout"],
                "balanced": pop.best_dual["balanced"],
                "params": pop.best_dual["params"],
                "algo_name": pop.best_dual.get("algo_name"),
                "sel_kind": pop.best_dual.get("sel_kind"),
                "metrics_evolve": pop.best_dual.get("metrics_e"),
                "metrics_holdout": pop.best_dual.get("metrics_h"),
            },
            "best_balanced": None if pop.best_bal is None else {
                "iter": pop.best_bal["iter"],
                "balanced": pop.best_bal["balanced"],
                "evolve": pop.best_bal["evolve"],
                "holdout": pop.best_bal["holdout"],
                "dual_min": pop.best_bal["dual_min"],
                "params": pop.best_bal["params"],
            },
            "diversity": stats,
            "elapsed_s": time.perf_counter() - t0,
            "v6_baseline_dual_min": 0.006008501357211069,
        }
        (OUT / "param_evolve_v7_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        (BEST_DIR / "best_program_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print("=== DONE v7 CREATIVE ===", flush=True)
        print(json.dumps(info.get("best_dual"), indent=2, default=str)[:2000], flush=True)
        print("=== DIVERSITY ===", flush=True)
        print(json.dumps(stats, indent=2), flush=True)
        if info.get("best_balanced"):
            print("=== BEST BAL ===", flush=True)
            print(json.dumps(info["best_balanced"], indent=2, default=str)[:1200], flush=True)
    else:
        print("No valid best found.")


if __name__ == "__main__":
    main()
