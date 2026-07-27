#!/usr/bin/env python
"""v5: dual_min-primary search + agree-gate + flex-adaptive half + Pareto mix.

Primary fitness = min(evolve, holdout)  (generalization)
Secondary = balanced for reporting.
Seeds from v4 dual/bal champs + new structure knobs.
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
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import importlib.util

spec_e = importlib.util.spec_from_file_location("evaluator", EX / "evaluator.py")
ev = importlib.util.module_from_spec(spec_e)
spec_e.loader.exec_module(ev)

OUT = EX / "openevolve_output_grok"
TRIAL = EX / "_param_trial_v5.py"

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
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (theta / (2.0 * np.sin(theta) + 1e-12))

def _so3_exp(w):
    theta = float(np.linalg.norm(w))
    if theta < 1e-10:
        return np.eye(3)
    k = w / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

def _local_scale(ca0, ca1, m, fracs, half):
    L, Ti = ca0.shape[0], len(fracs)
    out = np.empty((Ti, L, 3))
    for i in range(L):
        lo, hi = max(0, i - half), min(L, i + half + 1)
        sel = np.arange(lo, hi)
        sel = sel[m[sel]]
        if sel.size < 3:
            for ti, s in enumerate(fracs):
                out[ti, i] = (1.0 - s) * ca0[i] + s * ca1[i]
            continue
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0, mu1 = ca0[sel].mean(0), ca1[sel].mean(0)
        om = _so3_log(R)
        res = ca1[i] - ((ca0[i] - mu0) @ R.T + mu1)
        for ti, s in enumerate(fracs):
            s = float(s)
            Rs = _so3_exp(om * s)
            mus = (1.0 - s) * mu0 + s * mu1
            out[ti, i] = (ca0[i] - mu0) @ Rs.T + mus + s * res
    return out

def _local_flex(ca0, ca1, m, fracs, half_lo, half_hi, res_g):
    """Per-residue half from residual magnitude."""
    L, Ti = ca0.shape[0], len(fracs)
    rmag = np.linalg.norm(res_g, axis=1)
    r = rmag / (rmag.max() + 1e-8)
    half_i = (half_hi - (half_hi - half_lo) * r).astype(int)
    half_i = np.clip(half_i, half_lo, half_hi)
    out = np.empty((Ti, L, 3))
    for i in range(L):
        half = int(half_i[i])
        lo, hi = max(0, i - half), min(L, i + half + 1)
        sel = np.arange(lo, hi)
        sel = sel[m[sel]]
        if sel.size < 3:
            for ti, s in enumerate(fracs):
                out[ti, i] = (1.0 - s) * ca0[i] + s * ca1[i]
            continue
        R = _kabsch_R(ca0[sel], ca1[sel])
        mu0, mu1 = ca0[sel].mean(0), ca1[sel].mean(0)
        om = _so3_log(R)
        res = ca1[i] - ((ca0[i] - mu0) @ R.T + mu1)
        for ti, s in enumerate(fracs):
            s = float(s)
            Rs = _so3_exp(om * s)
            mus = (1.0 - s) * mu0 + s * mu1
            out[ti, i] = (ca0[i] - mu0) @ Rs.T + mus + s * res
    return out

def predict_interior(ca_start, ca_end, fracs, mask):
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    fracs = np.asarray(fracs, dtype=np.float64)
    p = PARAMS
    L = ca0.shape[0]
    Ti = len(fracs)
    half = max(2, int(p.get("local_half", 16)))
    half_s = max(2, int(p.get("half_s", 4)))
    half_lo = max(4, int(p.get("half_lo", 8)))
    half_hi = max(half_lo + 2, int(p.get("half_hi", 20)))
    s_gate = float(p.get("s_gate", 0.0))
    agree_str = float(p.get("agree_str", 0.0))
    use_flex = float(p.get("use_flex", 0.0))
    mag_mid = float(p.get("mag_mid", 0.0))
    mode_mid = float(p.get("mode_mid_boost", 0.0))
    w_ms_s = float(p.get("w_ms_small", 0.0))

    lin = np.empty((Ti, L, 3))
    for i, s in enumerate(fracs):
        s = float(s)
        lin[i] = (1.0 - s) * ca0 + s * ca1

    res_g = np.zeros((L, 3))
    screw = lin.copy()
    if m.sum() >= 3:
        R = _kabsch_R(ca0[m], ca1[m])
        mu0 = ca0[m].mean(0)
        mu1 = ca1[m].mean(0)
        omega = _so3_log(R)
        res_g = ca1 - ((ca0 - mu0) @ R.T + mu1)
        mag = np.linalg.norm(res_g, axis=1, keepdims=True)
        u = np.divide(res_g, mag + 1e-12)
        for i, s in enumerate(fracs):
            s = float(s)
            Rs = _so3_exp(omega * s)
            mus = (1.0 - s) * mu0 + s * mu1
            rigid = (ca0 - mu0) @ Rs.T + mus
            mag_s = (s + mag_mid * np.sin(np.pi * s)) * mag
            screw[i] = rigid + mag_s * u

    # local: fixed half vs flex-adaptive
    if use_flex > 0.5 and m.sum() >= 3:
        loc = _local_flex(ca0, ca1, m, fracs, half_lo, half_hi, res_g)
    else:
        loc_m = _local_scale(ca0, ca1, m, fracs, half)
        if w_ms_s > 0.02:
            loc_s = _local_scale(ca0, ca1, m, fracs, half_s)
            loc = (1.0 - w_ms_s) * loc_m + w_ms_s * loc_s
        else:
            loc = loc_m

    # agree-gate: boost local where local rigid endpoint differs from global rigid
    w_loc_base = float(p.get("w_local", 0.15))
    w_loc_i = np.full(L, w_loc_base, dtype=np.float64)
    if agree_str > 1e-6 and m.sum() >= 3:
        R = _kabsch_R(ca0[m], ca1[m])
        mu0 = ca0[m].mean(0)
        mu1 = ca1[m].mean(0)
        agree = np.zeros(L)
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
            glob_rig = (ca0[i] - mu0) @ R.T + mu1
            loc_rig = (ca0[i] - m0) @ Rl.T + m1
            agree[i] = float(np.linalg.norm(loc_rig - glob_rig))
        amax = float(agree.max()) + 1e-8
        an = agree / amax
        w_loc_i = w_loc_base + agree_str * an
        for _ in range(2):
            w2 = w_loc_i.copy()
            w2[1:-1] = 0.25 * w_loc_i[:-2] + 0.5 * w_loc_i[1:-1] + 0.25 * w_loc_i[2:]
            w_loc_i = w2
        w_loc_i = w_loc_i * m.astype(np.float64)

    # mode path
    mode_path = lin.copy()
    nmodes = max(0, int(p.get("n_modes", 0)))
    w_mode = float(p.get("w_mode", 0.0))
    if nmodes > 0 and w_mode > 0.02 and m.sum() >= 6:
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
                s = float(s)
                mid = mode_mid * np.sin(np.pi * s)
                delta = np.zeros((L, 3))
                delta[idxm] = (s + mid) * recon + s * miss
                mode_path[i] = ca0 + delta
        except np.linalg.LinAlgError:
            pass

    w_lin = float(p.get("w_lin", 0.1))
    w_screw = float(p.get("w_screw", 0.6))
    # if agree-gate active, per-residue weights; else static
    raw_static = np.array([w_lin, w_screw, w_loc_base, w_mode], dtype=np.float64)
    raw_static = np.maximum(raw_static, 0.0)
    if raw_static.sum() < 1e-12:
        raw_static[0] = 1.0
    raw_static = raw_static / raw_static.sum()

    out = np.empty((Ti, L, 3))
    for i, s in enumerate(fracs):
        s = float(s)
        mid = np.sin(np.pi * s)
        if agree_str > 1e-6:
            # per-residue: w_loc_i, redistribute remaining to lin/screw/mode
            wl = np.clip(w_loc_i, 0.0, 0.55)
            rem = np.clip(1.0 - wl, 0.05, 1.0)
            # base shares among non-local
            base = np.array([w_lin, w_screw, max(w_mode, 0.0)], dtype=np.float64)
            base = np.maximum(base, 0.0)
            if base.sum() < 1e-12:
                base = np.array([0.2, 0.8, 0.0])
            base = base / base.sum()
            # s-gate on non-local: push lin -> screw
            b = base.copy()
            take = min(b[0], s_gate * mid * 0.25)
            b[0] -= take
            b[1] += take
            b = b / (b.sum() + 1e-12)
            wn = rem * b[0]
            ws = rem * b[1]
            wm = rem * b[2]
            out[i] = (
                wn[:, None] * lin[i]
                + ws[:, None] * screw[i]
                + wl[:, None] * loc[i]
                + wm[:, None] * mode_path[i]
            )
        else:
            w = raw_static.copy()
            take = min(w[0], s_gate * mid * 0.25)
            w[0] -= take
            w[1] += take * 0.65
            w[2] += take * 0.35
            w = w / (w.sum() + 1e-12)
            out[i] = w[0] * lin[i] + w[1] * screw[i] + w[2] * loc[i] + w[3] * mode_path[i]
    return out

def run_interpolation(ca_start, ca_end, fracs, mask):
    return predict_interior(ca_start, ca_end, fracs, mask)
'''


def materialize(params: dict, path: Path) -> None:
    src = PROG.replace("PARAMS = {}", "PARAMS = " + repr(params), 1)
    path.write_text(src, encoding="utf-8")


def balanced(es: float, hs: float) -> float:
    return 0.5 * es + 0.5 * hs - 0.35 * max(0.0, -hs)


def dual_score(es: float, hs: float) -> float:
    """Primary: min(e,h) with soft bonus when both positive and close."""
    mn = min(es, hs)
    if es > 0 and hs > 0:
        # reward balance: higher when e≈h
        gap = abs(es - hs)
        return mn + 0.15 * mn * (1.0 / (1.0 + 20.0 * gap))
    return mn - 0.5 * max(0.0, -min(es, hs))


def clip(x, a, b):
    return a if x < a else b if x > b else x


def sample(rng: random.Random) -> dict:
    raw = [rng.random() for _ in range(4)]
    raw[3] *= 0.5  # mode smaller
    s = sum(raw) + 1e-12
    w = [x / s for x in raw]
    return {
        "w_lin": w[0],
        "w_screw": w[1],
        "w_local": w[2],
        "w_mode": w[3],
        "local_half": rng.randint(12, 22),
        "half_s": rng.choice([3, 4, 5, 6]),
        "half_lo": rng.choice([6, 8, 10]),
        "half_hi": rng.choice([16, 18, 20, 22]),
        "w_ms_small": rng.choice([0.0, 0.0, 0.0, 0.08, 0.12, 0.18]),
        "s_gate": rng.choice([0.0, 0.0, 0.1, 0.15, 0.2, 0.3]),
        "agree_str": rng.choice([0.0, 0.0, 0.0, 0.08, 0.15, 0.25]),
        "use_flex": rng.choice([0.0, 0.0, 0.0, 1.0]),
        "mag_mid": rng.choice([0.0, 0.0, 0.03, 0.06, 0.1]),
        "n_modes": rng.choice([0, 1, 1, 1, 2]),
        "mode_mid_boost": rng.choice([0.0, 0.05, 0.1, 0.12, 0.15]),
    }


def mutate(p: dict, rng: random.Random, scale: float = 0.3) -> dict:
    q = copy.deepcopy(p)
    keys = list(q.keys())
    for k in rng.sample(keys, k=rng.randint(1, 5)):
        if k in ("local_half", "half_s", "half_lo", "half_hi", "n_modes"):
            step = rng.choice([-2, -1, 1, 2])
            lohi = {
                "local_half": (8, 24),
                "half_s": (2, 10),
                "half_lo": (4, 14),
                "half_hi": (14, 26),
                "n_modes": (0, 3),
            }[k]
            q[k] = int(clip(int(q[k]) + step, lohi[0], lohi[1]))
        elif k == "use_flex":
            q[k] = 1.0 if rng.random() < 0.3 else 0.0
        elif k.startswith("w_"):
            q[k] = float(max(0.0, q[k] + rng.gauss(0, 0.07 * scale)))
        else:
            q[k] = float(clip(float(q[k]) + rng.gauss(0, 0.07 * scale), 0.0, 1.0))
    ws = ["w_lin", "w_screw", "w_local", "w_mode"]
    s = sum(max(0.0, q[k]) for k in ws) + 1e-12
    for k in ws:
        q[k] = max(0.0, q[k]) / s
    if q["half_hi"] <= q["half_lo"]:
        q["half_hi"] = int(q["half_lo"]) + 4
    return q


SEEDS = [
    # linear
    dict(w_lin=1, w_screw=0, w_local=0, w_mode=0, local_half=16, half_s=4, half_lo=8, half_hi=20,
         w_ms_small=0, s_gate=0, agree_str=0, use_flex=0, mag_mid=0, n_modes=0, mode_mid_boost=0),
    # v4 balanced
    dict(w_lin=0.122, w_screw=0.675, w_local=0.145, w_mode=0.057, local_half=13, half_s=5,
         half_lo=8, half_hi=20, w_ms_small=0, s_gate=0, agree_str=0, use_flex=0, mag_mid=0,
         n_modes=1, mode_mid_boost=0.113),
    # v4 dual_min
    dict(w_lin=0.107, w_screw=0.661, w_local=0.175, w_mode=0.056, local_half=20, half_s=3,
         half_lo=8, half_hi=22, w_ms_small=0.134, s_gate=0.147, agree_str=0, use_flex=0, mag_mid=0,
         n_modes=1, mode_mid_boost=0.113),
    # dual + mild agree
    dict(w_lin=0.11, w_screw=0.65, w_local=0.16, w_mode=0.05, local_half=18, half_s=4,
         half_lo=8, half_hi=20, w_ms_small=0.1, s_gate=0.15, agree_str=0.15, use_flex=0, mag_mid=0,
         n_modes=1, mode_mid_boost=0.11),
    # flex half
    dict(w_lin=0.11, w_screw=0.66, w_local=0.18, w_mode=0.05, local_half=16, half_s=4,
         half_lo=8, half_hi=20, w_ms_small=0, s_gate=0.15, agree_str=0, use_flex=1, mag_mid=0.03,
         n_modes=1, mode_mid_boost=0.1),
    # mag mid residual
    dict(w_lin=0.12, w_screw=0.68, w_local=0.15, w_mode=0.05, local_half=16, half_s=4,
         half_lo=8, half_hi=20, w_ms_small=0.08, s_gate=0.1, agree_str=0.08, use_flex=0, mag_mid=0.06,
         n_modes=1, mode_mid_boost=0.1),
    # higher dual-ish local
    dict(w_lin=0.15, w_screw=0.55, w_local=0.25, w_mode=0.05, local_half=18, half_s=4,
         half_lo=7, half_hi=22, w_ms_small=0.12, s_gate=0.2, agree_str=0.2, use_flex=1, mag_mid=0,
         n_modes=1, mode_mid_boost=0.12),
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--screen-windows", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.environ["HUMAN_EVOLVE_DEVICE"] = args.device
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "param_evolve_v5_log.jsonl"
    rng = random.Random(args.seed)
    best_dual = None
    best_bal = None
    t0 = time.perf_counter()

    def eval_file_score(path, n_windows, split):
        return ev._score(str(path), n_windows=n_windows, split=split, timeout=180)

    def consider(params, it, force_full=False):
        nonlocal best_dual, best_bal
        materialize(params, TRIAL)
        scr = eval_file_score(TRIAL, args.screen_windows, "evolve")
        screen_sc = float(scr.get("combined_score", -1))
        # more permissive full eval for dual search
        thr = -0.005
        if best_dual is not None:
            thr = best_dual.get("screen", -9) - 0.008
        do_full = force_full or best_dual is None or screen_sc > thr
        if not do_full:
            return screen_sc, None
        full_e = eval_file_score(TRIAL, None, "evolve")
        full_h = eval_file_score(TRIAL, None, "holdout")
        es = float(full_e.get("combined_score", -1))
        hs = float(full_h.get("combined_score", -1))
        bal = balanced(es, hs)
        dscore = dual_score(es, hs)
        dmin = min(es, hs)
        rec = {
            "params": copy.deepcopy(params),
            "screen": screen_sc,
            "evolve": es,
            "holdout": hs,
            "balanced": bal,
            "dual_score": dscore,
            "dual_min": dmin,
            "metrics_e": full_e,
            "metrics_h": full_h,
            "iter": it,
        }
        if best_dual is None or dscore > best_dual["dual_score"] + 1e-9:
            best_dual = rec
            print(
                f"[{it:4d}] NEW DUAL  dscore={dscore:+.5f} min={dmin:+.5f}  "
                f"e={es:+.5f} h={hs:+.5f}  half={params.get('local_half')} "
                f"sg={params.get('s_gate'):.2f} agr={params.get('agree_str'):.2f} "
                f"flex={params.get('use_flex'):.0f} loc={params.get('w_local'):.2f}",
                flush=True,
            )
        if best_bal is None or bal > best_bal["balanced"] + 1e-9:
            best_bal = rec
            print(
                f"[{it:4d}] NEW BAL   bal={bal:+.5f} e={es:+.5f} h={hs:+.5f}",
                flush=True,
            )
        return screen_sc, rec

    print(f"=== param evolve v5 dual-primary: {args.iterations} iters device={args.device} ===", flush=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        for i, seed in enumerate(SEEDS):
            sc, rec = consider(seed, it=-(i + 1), force_full=True)
            row = {"iter": -(i + 1), "seed": True, "screen": sc}
            if rec:
                row.update({k: rec[k] for k in ("evolve", "holdout", "balanced", "dual_score", "dual_min", "params")})
            logf.write(json.dumps(row) + "\n")
            logf.flush()

        for it in range(args.iterations):
            r = rng.random()
            if r < 0.5 and best_dual is not None:
                params = mutate(best_dual["params"], rng, scale=0.25 + 0.4 * rng.random())
            elif r < 0.75 and best_bal is not None:
                params = mutate(best_bal["params"], rng, scale=0.35)
            else:
                params = sample(rng)

            sc, rec = consider(params, it=it)
            row = {"iter": it, "screen": sc, "params": params}
            if rec:
                row.update({
                    "evolve": rec["evolve"],
                    "holdout": rec["holdout"],
                    "balanced": rec["balanced"],
                    "dual_score": rec["dual_score"],
                    "dual_min": rec["dual_min"],
                })
            else:
                row.update({"evolve": None, "holdout": None, "balanced": None,
                            "dual_score": None, "dual_min": None})
            logf.write(json.dumps(row) + "\n")
            logf.flush()

            if (it + 1) % 25 == 0:
                d = best_dual["dual_min"] if best_dual else None
                b = best_bal["balanced"] if best_bal else None
                print(
                    f"  ... {it+1}/{args.iterations}  best_dual_min={d}  best_bal={b}  "
                    f"elapsed={time.perf_counter()-t0:.0f}s",
                    flush=True,
                )

    BEST = OUT / "best"
    BEST.mkdir(parents=True, exist_ok=True)
    info = {
        "source": "param_evolve_v5",
        "iterations": args.iterations,
        "device": args.device,
        "elapsed_s": time.perf_counter() - t0,
        "best_dual": None,
        "best_balanced": None,
    }
    if best_dual is not None:
        materialize(best_dual["params"], BEST / "best_program_v5_dual.py")
        materialize(best_dual["params"], BEST / "best_program.py")
        materialize(best_dual["params"], EX / "candidates" / "29_v5_dual_min.py")
        info["best_dual"] = {
            "iter": best_dual["iter"],
            "dual_score": best_dual["dual_score"],
            "dual_min": best_dual["dual_min"],
            "evolve": best_dual["evolve"],
            "holdout": best_dual["holdout"],
            "balanced": best_dual["balanced"],
            "params": best_dual["params"],
            "metrics_evolve": best_dual["metrics_e"],
            "metrics_holdout": best_dual["metrics_h"],
        }
    if best_bal is not None:
        materialize(best_bal["params"], BEST / "best_program_v5_balanced.py")
        info["best_balanced"] = {
            "iter": best_bal["iter"],
            "balanced": best_bal["balanced"],
            "evolve": best_bal["evolve"],
            "holdout": best_bal["holdout"],
            "dual_min": best_bal["dual_min"],
            "params": best_bal["params"],
        }
    (OUT / "param_evolve_v5_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (BEST / "best_program_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("=== DONE v5 ===", flush=True)
    print(json.dumps(info.get("best_dual"), indent=2, default=str)[:2000], flush=True)
    if info.get("best_balanced"):
        print("=== BEST BAL ===", flush=True)
        print(json.dumps(info["best_balanced"], indent=2, default=str)[:1200], flush=True)


if __name__ == "__main__":
    main()
