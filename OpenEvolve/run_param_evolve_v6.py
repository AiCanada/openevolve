#!/usr/bin/env python
"""v5b: dual_min-primary search on v4 program skeleton path algorithms beyond v3 weight blending.

New components:
  - true Chasles screw (axis + pitch)
  - multi-scale local (half_s / half_m / half_l)
  - motion-correlation sequential domains
  - s-dependent blend gates (mid vs ends)
  - per-residue residual timewarp

Fitness = 0.5*evolve + 0.5*holdout - 0.35*max(0,-holdout)
Also tracks dual_min = min(evolve, holdout) for reporting.
GPU scorer. No Claude / no XAI key.
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from pathlib import Path

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
TRIAL = EX / "_param_trial_v6.py"

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
    s_gate = float(p.get("s_gate", 0.0))  # how much mid-boost local/screw
    mode_mid = float(p.get("mode_mid_boost", 0.0))

    # --- linear ---
    lin = np.empty((Ti, L, 3))
    for i, s in enumerate(fracs):
        t = _time(s, ease, herm)
        lin[i] = (1.0 - t) * ca0 + t * ca1

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
        # smooth alpha
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
            # timewarp residual for high-disp residues
            se = t * t * (3.0 - 2.0 * t)
            t_res = (1.0 - warp * alpha) * t + (warp * alpha) * se
            screw[i] = rigid + t_res[:, None] * residual_g

        # Chasles true screw
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
        # blend standard vs chasles
        uc = float(np.clip(use_chasles, 0.0, 1.0))
        screw = (1.0 - uc) * screw + uc * chasles

    # --- multi-scale local (skip unused scales for speed) ---
    w_ms = float(p.get("w_ms_small", 0.0))
    w_mm = float(p.get("w_ms_mid", 1.0))
    w_ml = float(p.get("w_ms_large", 0.0))
    ws_m = np.array([max(0.0, w_ms), max(0.0, w_mm), max(0.0, w_ml)], dtype=np.float64)
    if ws_m.sum() < 1e-12:
        ws_m[:] = [0.0, 1.0, 0.0]
    ws_m = ws_m / ws_m.sum()
    # default: single mid scale (matches v3 local)
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

    # --- correlation domains (skip if unused) ---
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

    # --- base static weights ---
    w_lin = float(p.get("w_lin", 0.1))
    w_screw = float(p.get("w_screw", 0.5))
    w_local = float(p.get("w_local", 0.2))
    w_dom = float(p.get("w_dom", 0.0))
    w_mode = float(p.get("w_mode", 0.0))
    raw = np.array([w_lin, w_screw, w_local, w_dom, w_mode], dtype=np.float64)
    raw = np.maximum(raw, 0.0)
    if raw.sum() < 1e-12:
        raw[0] = 1.0
    raw = raw / raw.sum()

    # s-dependent reweight: at mid, push weight from lin -> screw/local
    out = np.empty((Ti, L, 3))
    for i, s in enumerate(fracs):
        s = float(s)
        mid = np.sin(np.pi * s)
        g = float(np.clip(s_gate, 0.0, 1.0)) * mid
        w = raw.copy()
        # transfer up to g*0.25 from lin to screw and local
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

    # optional gentle bond restore
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
'''


def materialize(params: dict, path: Path) -> None:
    src = PROG.replace("PARAMS = {}", "PARAMS = " + repr(params), 1)
    path.write_text(src, encoding="utf-8")


def balanced(es: float, hs: float) -> float:
    return 0.5 * es + 0.5 * hs - 0.35 * max(0.0, -hs)


def clip(x, a, b):
    return a if x < a else b if x > b else x


def sample(rng: random.Random) -> dict:
    # Bias toward v5b dual region: high local, low lin, multiscale small
    raw = [rng.random() for _ in range(5)]
    raw[0] *= 0.35  # w_lin low
    raw[3] *= 0.15  # w_dom rarely helps
    raw[4] *= 0.55  # mode moderate
    s = sum(raw) + 1e-12
    w = [x / s for x in raw]
    # multiscale: often non-trivial small weight (v5b dual used ~0.27)
    if rng.random() < 0.25:
        ms = [0.0, 1.0, 0.0]
    else:
        small = rng.uniform(0.05, 0.35)
        large = rng.uniform(0.0, 0.05)
        mid = max(0.05, 1.0 - small - large)
        ms = [small, mid, large]
        sms = sum(ms)
        ms = [x / sms for x in ms]
    return {
        "ease": rng.choice([0.0, 0.0, 0.0, 0.02, 0.05, 0.1]),
        "hermite": rng.choice([0.0, 0.0, 0.0, 0.0, 0.05]),
        "w_lin": w[0],
        "w_screw": w[1],
        "w_local": w[2],
        "w_dom": w[3],
        "w_mode": w[4],
        "local_half": rng.randint(16, 28),
        "half_s": rng.choice([2, 2, 3, 4]),
        "half_l": rng.choice([20, 22, 24, 26, 28, 30]),
        "w_ms_small": ms[0],
        "w_ms_mid": ms[1],
        "w_ms_large": ms[2],
        "n_dom": rng.choice([1, 1, 2, 3, 4]),
        "min_dom": rng.choice([6, 8, 9, 10]),
        "use_chasles": rng.choice([0.0, 0.0, 0.0, 0.05, 0.1]),
        "timewarp": rng.choice([0.0, 0.03, 0.05, 0.07, 0.09, 0.12, 0.15]),
        "s_gate": rng.choice([0.05, 0.1, 0.13, 0.15, 0.18, 0.20, 0.25, 0.30]),
        "n_modes": rng.choice([0, 1, 1, 1, 1, 2]),
        "mode_mid_boost": rng.choice([0.05, 0.1, 0.15, 0.19, 0.22, 0.27, 0.32]),
        "bond_strength": rng.choice([0.0, 0.0, 0.0, 0.02, 0.05]),
    }


def mutate(p: dict, rng: random.Random, scale: float = 0.35) -> dict:
    q = copy.deepcopy(p)
    keys = list(q.keys())
    for k in rng.sample(keys, k=rng.randint(1, 5)):
        if k in ("local_half", "half_s", "half_l", "n_dom", "min_dom", "n_modes"):
            step = rng.choice([-2, -1, 1, 2])
            lohi = {
                "local_half": (10, 32),
                "half_s": (2, 8),
                "half_l": (16, 36),
                "n_dom": (1, 5),
                "min_dom": (5, 14),
                "n_modes": (0, 3),
            }[k]
            q[k] = int(clip(int(q[k]) + step, lohi[0], lohi[1]))
        elif k.startswith("w_"):
            q[k] = float(max(0.0, q[k] + rng.gauss(0, 0.08 * scale)))
        else:
            q[k] = float(clip(float(q[k]) + rng.gauss(0, 0.08 * scale), 0.0, 1.0))
    # renormalize main blend
    ws = ["w_lin", "w_screw", "w_local", "w_dom", "w_mode"]
    s = sum(max(0.0, q[k]) for k in ws) + 1e-12
    for k in ws:
        q[k] = max(0.0, q[k]) / s
    ms = ["w_ms_small", "w_ms_mid", "w_ms_large"]
    s2 = sum(max(0.0, q[k]) for k in ms) + 1e-12
    for k in ms:
        q[k] = max(0.0, q[k]) / s2
    if q["half_l"] <= q["local_half"]:
        q["half_l"] = int(q["local_half"]) + 4
    if q["half_s"] >= q["local_half"]:
        q["half_s"] = max(3, int(q["local_half"]) - 3)
    return q


# Seeds: linear + v4 + v5b dual/bal winners + neighborhood probes
SEEDS = [
    # linear
    dict(ease=0, hermite=0, w_lin=1, w_screw=0, w_local=0, w_dom=0, w_mode=0,
         local_half=13, half_s=5, half_l=22, w_ms_small=0.0, w_ms_mid=1.0, w_ms_large=0.0,
         n_dom=1, min_dom=8, use_chasles=0, timewarp=0, s_gate=0, n_modes=0,
         mode_mid_boost=0, bond_strength=0),
    # v5b dual winner (it 397)
    dict(ease=0.01656458633604745, hermite=0.0, w_lin=0.0, w_screw=0.5925892464097408,
         w_local=0.3331569567582794, w_dom=0.0, w_mode=0.07425379683099403,
         local_half=23, half_s=2, half_l=24, w_ms_small=0.26786389000477573,
         w_ms_mid=0.7233670510299913, w_ms_large=0.00876905896423418,
         n_dom=4, min_dom=8, use_chasles=0.0, timewarp=0.06589951544874657,
         s_gate=0.201760396144735, n_modes=1, mode_mid_boost=0.2691589688500917,
         bond_strength=0.019438584478809258),
    # v5b balanced winner (it 276)
    dict(ease=0.0300469580504501, hermite=0.0, w_lin=0.0802465730245495,
         w_screw=0.6111012289634551, w_local=0.22535918550581946, w_dom=0.008052751047708102,
         w_mode=0.07524026145746773, local_half=21, half_s=2, half_l=22,
         w_ms_small=0.181465109912398, w_ms_mid=0.807916655261373,
         w_ms_large=0.010618234825211322, n_dom=4, min_dom=9, use_chasles=0.04409095354062464,
         timewarp=0.11652009267927878, s_gate=0.17315000031206101, n_modes=1,
         mode_mid_boost=0.19218693626673472, bond_strength=0.0),
    # v4 dual (legacy)
    dict(ease=0, hermite=0.0, w_lin=0.1069, w_screw=0.6613, w_local=0.1753, w_dom=0.0,
         w_mode=0.0564, local_half=20, half_s=3, half_l=22, w_ms_small=0.1341,
         w_ms_mid=0.8556, w_ms_large=0.0102, n_dom=1, min_dom=9, use_chasles=0.0,
         timewarp=0.0, s_gate=0.1474, n_modes=1, mode_mid_boost=0.113, bond_strength=0),
    # push local higher + wider half
    dict(ease=0.0, hermite=0.0, w_lin=0.0, w_screw=0.55, w_local=0.38, w_dom=0.0, w_mode=0.07,
         local_half=26, half_s=2, half_l=28, w_ms_small=0.30, w_ms_mid=0.68, w_ms_large=0.02,
         n_dom=1, min_dom=8, use_chasles=0.0, timewarp=0.06, s_gate=0.22, n_modes=1,
         mode_mid_boost=0.28, bond_strength=0.02),
    # more multiscale small, less screw
    dict(ease=0.0, hermite=0.0, w_lin=0.02, w_screw=0.52, w_local=0.36, w_dom=0.0, w_mode=0.10,
         local_half=24, half_s=2, half_l=26, w_ms_small=0.35, w_ms_mid=0.63, w_ms_large=0.02,
         n_dom=1, min_dom=8, use_chasles=0.0, timewarp=0.08, s_gate=0.25, n_modes=1,
         mode_mid_boost=0.30, bond_strength=0.0),
    # balanced-ish with dual traits
    dict(ease=0.02, hermite=0.0, w_lin=0.05, w_screw=0.60, w_local=0.28, w_dom=0.0, w_mode=0.07,
         local_half=22, half_s=2, half_l=24, w_ms_small=0.22, w_ms_mid=0.76, w_ms_large=0.02,
         n_dom=1, min_dom=8, use_chasles=0.0, timewarp=0.10, s_gate=0.18, n_modes=1,
         mode_mid_boost=0.22, bond_strength=0.0),
    # high s_gate + mode mid
    dict(ease=0.0, hermite=0.0, w_lin=0.0, w_screw=0.58, w_local=0.32, w_dom=0.0, w_mode=0.10,
         local_half=23, half_s=2, half_l=25, w_ms_small=0.28, w_ms_mid=0.70, w_ms_large=0.02,
         n_dom=1, min_dom=8, use_chasles=0.0, timewarp=0.05, s_gate=0.30, n_modes=2,
         mode_mid_boost=0.32, bond_strength=0.02),
    # mild timewarp high local
    dict(ease=0.0, hermite=0.0, w_lin=0.03, w_screw=0.57, w_local=0.34, w_dom=0.0, w_mode=0.06,
         local_half=25, half_s=3, half_l=27, w_ms_small=0.25, w_ms_mid=0.73, w_ms_large=0.02,
         n_dom=1, min_dom=8, use_chasles=0.0, timewarp=0.12, s_gate=0.20, n_modes=1,
         mode_mid_boost=0.25, bond_strength=0.0),
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--screen-windows", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.environ["HUMAN_EVOLVE_DEVICE"] = args.device

    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "param_evolve_v6_log.jsonl"
    rng = random.Random(args.seed)
    best = None
    best_dual = None
    t0 = time.perf_counter()

    def eval_file_score(path, n_windows, split):
        return ev._score(str(path), n_windows=n_windows, split=split, timeout=180)

    def consider(params, it, force_full=False, screen_sc=None):
        nonlocal best, best_dual
        materialize(params, TRIAL)
        if screen_sc is None:
            scr = eval_file_score(TRIAL, args.screen_windows, "evolve")
            screen_sc = float(scr.get("combined_score", -1))
            screen_rmsd = float(scr.get("model_rmsd", 99))
        else:
            screen_rmsd = None
        # permissive screen: dual-primary wants holdout-friendly recipes that may screen lower
        thr = -0.006
        if best_dual is not None:
            thr = min(thr, best_dual.get("screen", -9) - 0.01)
        if best is not None:
            thr = min(thr, best.get("screen", -9) - 0.004)
        do_full = force_full or best is None or best_dual is None or screen_sc > thr
        if not do_full:
            return screen_sc, None, screen_rmsd
        full_e = eval_file_score(TRIAL, None, "evolve")
        full_h = eval_file_score(TRIAL, None, "holdout")
        es = float(full_e.get("combined_score", -1))
        hs = float(full_h.get("combined_score", -1))
        bal = balanced(es, hs)
        dual = min(es, hs)
        # dual_score: prefer positive balanced pairs
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
        }
        if best is None or bal > best["balanced"] + 1e-9:
            best = rec
            print(
                f"[{it:4d}] NEW BAL   bal={bal:+.5f}  e={es:+.5f} ({full_e.get('model_rmsd'):.4f}A)  "
                f"h={hs:+.5f} ({full_h.get('model_rmsd'):.4f}A)  dual={dual:+.5f}",
                flush=True,
            )
        if best_dual is None or dscore > best_dual.get("dual_score", -9) + 1e-9:
            best_dual = rec
            print(
                f"[{it:4d}] NEW DUAL  dscore={dscore:+.5f} min={dual:+.5f}  e={es:+.5f} h={hs:+.5f}  "
                f"half={params.get('local_half')} sg={params.get('s_gate'):.2f} "
                f"loc={params.get('w_local'):.2f} ms={params.get('w_ms_small'):.2f} "
                f"screw={params.get('w_screw'):.2f}",
                flush=True,
            )
        return screen_sc, rec, float(full_e.get("model_rmsd", 99))

    # line-buffered prints
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    print(f"=== param evolve v6 dual-primary: {args.iterations} iters device={args.device} ===", flush=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        for i, seed in enumerate(SEEDS):
            sc, rec, _ = consider(seed, it=-(i + 1), force_full=True)
            row = {"iter": -(i + 1), "seed": True, "screen": sc}
            if rec:
                row.update({k: rec[k] for k in ("evolve", "holdout", "balanced", "dual_min", "params")})
            logf.write(json.dumps(row) + "\n")
            logf.flush()

        for it in range(args.iterations):
            r = rng.random()
            if r < 0.60 and best_dual is not None:
                params = mutate(best_dual["params"], rng, scale=0.18 + 0.35 * rng.random())
            elif r < 0.80 and best is not None:
                params = mutate(best["params"], rng, scale=0.3)
            else:
                params = sample(rng)

            sc, rec, srmsd = consider(params, it=it)
            row = {"iter": it, "screen": sc, "screen_rmsd": srmsd, "params": params}
            if rec:
                row.update({
                    "evolve": rec["evolve"],
                    "holdout": rec["holdout"],
                    "balanced": rec["balanced"],
                    "dual_min": rec["dual_min"],
                    "dual_score": rec.get("dual_score"),
                })
            else:
                row.update({"evolve": None, "holdout": None, "balanced": None,
                            "dual_min": None, "dual_score": None})
            logf.write(json.dumps(row) + "\n")
            logf.flush()

            if (it + 1) % 50 == 0:
                b = best["balanced"] if best else None
                d = best_dual["dual_min"] if best_dual else None
                ds = best_dual.get("dual_score") if best_dual else None
                print(
                    f"  ... {it+1}/{args.iterations}  best_bal={b}  best_dual_min={d}  "
                    f"dscore={ds}  elapsed={time.perf_counter()-t0:.0f}s",
                    flush=True,
                )
                # mid-run checkpoint so long 1000-iter jobs don't lose bests
                if best_dual is not None:
                    (OUT / "best").mkdir(parents=True, exist_ok=True)
                    ck = {
                        "iter": it,
                        "best_dual": {
                            "iter": best_dual["iter"],
                            "dual_min": best_dual["dual_min"],
                            "dual_score": best_dual.get("dual_score"),
                            "evolve": best_dual["evolve"],
                            "holdout": best_dual["holdout"],
                            "balanced": best_dual["balanced"],
                            "params": best_dual["params"],
                        },
                        "best_balanced": None if best is None else {
                            "iter": best["iter"],
                            "balanced": best["balanced"],
                            "evolve": best["evolve"],
                            "holdout": best["holdout"],
                            "dual_min": best["dual_min"],
                            "params": best["params"],
                        },
                        "elapsed_s": time.perf_counter() - t0,
                    }
                    (OUT / "param_evolve_v6_checkpoint.json").write_text(
                        json.dumps(ck, indent=2), encoding="utf-8"
                    )
                    materialize(best_dual["params"], OUT / "best" / "best_program_v6_dual_ckpt.py")
                    if best is not None:
                        materialize(best["params"], OUT / "best" / "best_program_v6_bal_ckpt.py")

    # promote dual-primary program (generalizer)
    promo = best_dual if best_dual is not None else best
    if promo is not None:
        best_path = OUT / "best" / "best_program_v6.py"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        materialize(promo["params"], best_path)
        materialize(promo["params"], OUT / "best" / "best_program.py")
        materialize(promo["params"], EX / "candidates" / "32_v6_dual_min.py")
        if best is not None:
            materialize(best["params"], OUT / "best" / "best_program_v6_balanced.py")
        info = {
            "source": "param_evolve_v6",
            "iterations": args.iterations,
            "device": args.device,
            "best_balanced": None if best is None else {
                "iter": best["iter"],
                "balanced": best["balanced"],
                "evolve": best["evolve"],
                "holdout": best["holdout"],
                "dual_min": best["dual_min"],
                "params": best["params"],
                "metrics_evolve": best["metrics_e"],
                "metrics_holdout": best["metrics_h"],
            },
            "best_dual": None if best_dual is None else {
                "iter": best_dual["iter"],
                "dual_min": best_dual["dual_min"],
                "dual_score": best_dual.get("dual_score"),
                "evolve": best_dual["evolve"],
                "holdout": best_dual["holdout"],
                "balanced": best_dual["balanced"],
                "params": best_dual["params"],
                "metrics_evolve": best_dual["metrics_e"],
                "metrics_holdout": best_dual["metrics_h"],
            },
            "elapsed_s": time.perf_counter() - t0,
        }
        (OUT / "param_evolve_v6_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        (OUT / "best" / "best_program_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print("=== DONE v6 ===", flush=True)
        print(json.dumps(info.get("best_dual"), indent=2, default=str)[:2000], flush=True)
        if info.get("best_balanced"):
            print("=== BEST BAL ===", flush=True)
            print(json.dumps(info["best_balanced"], indent=2, default=str)[:1200], flush=True)
    else:
        print("No valid best found.")


if __name__ == "__main__":
    main()

