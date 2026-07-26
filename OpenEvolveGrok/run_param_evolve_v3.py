#!/usr/bin/env python
"""v3: intensive local search + new components for better path algorithms.

Starts from v2 best, adds multi-hinge / residual-PCA / soft-local weights.
Fitness = 0.5*evolve + 0.5*holdout - penalty for holdout regression.
GPU scorer. No Claude / no XAI key.
"""
from __future__ import annotations

import copy
import json
import os
import random
import re
import shutil
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
TRIAL = EX / "_param_trial_v3.py"

# ---------------------------------------------------------------------------
# Program source (self-contained, written each trial)
# ---------------------------------------------------------------------------
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
    nseg = max(1, int(p.get("nseg", 3)))
    nhinge = max(1, int(p.get("n_hinge", 1)))
    nmodes = max(0, int(p.get("n_modes", 1)))
    soft = float(p.get("soft_local", 0.0))  # soft weighting by displacement

    # --- components ---
    lin = np.empty((Ti, L, 3))
    for i, s in enumerate(fracs):
        t = _time(s, ease, herm)
        lin[i] = (1.0 - t) * ca0 + t * ca1

    # global screw + residual
    screw = np.empty((Ti, L, 3))
    if m.sum() >= 3:
        w = m.astype(np.float64)
        sw = w.sum()
        mu0 = (w[:, None] * ca0).sum(0) / sw
        mu1 = (w[:, None] * ca1).sum(0) / sw
        R = _kabsch_R(ca0[m], ca1[m])
        omega = _so3_log(R)
        residual = ca1 - ((ca0 - mu0) @ R.T + mu1)
        rmag = np.linalg.norm(residual, axis=1)
        rmag = rmag / (rmag.max() + 1e-8)
        beta = float(p.get("res_mid_boost", 0.0))
        for i, s in enumerate(fracs):
            t = _time(s, ease, herm)
            Rs = _so3_exp(omega * t)
            mus = (1.0 - t) * mu0 + t * mu1
            rigid = (ca0 - mu0) @ Rs.T + mus
            prog = t + beta * np.sin(np.pi * s) * rmag
            screw[i] = rigid + prog[:, None] * residual
    else:
        screw = lin.copy()

    # multi-segment
    multi = screw.copy()
    idx = np.where(m)[0]
    if idx.size >= nseg * 3:
        cuts = np.linspace(0, idx.size, nseg + 1).astype(int)
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < 3:
                continue
            sel = idx[a:b]
            R = _kabsch_R(ca0[sel], ca1[sel])
            mu0s, mu1s = ca0[sel].mean(0), ca1[sel].mean(0)
            om = _so3_log(R)
            res = ca1[sel] - ((ca0[sel] - mu0s) @ R.T + mu1s)
            for i, s in enumerate(fracs):
                t = _time(s, ease, herm)
                Rs = _so3_exp(om * t)
                mus = (1.0 - t) * mu0s + t * mu1s
                multi[i, sel] = (ca0[sel] - mu0s) @ Rs.T + mus + t * res

    # local window screw
    loc = np.empty((Ti, L, 3))
    disp = np.linalg.norm(ca1 - ca0, axis=1)
    for i in range(L):
        lo, hi = max(0, i - half), min(L, i + half + 1)
        sel = np.arange(lo, hi)
        sel = sel[m[sel]]
        if sel.size < 3:
            loc[:, i] = lin[:, i]
            continue
        if soft > 1e-6:
            # soft weights by inverse distance in sequence * displacement similarity
            center = i
            ww = np.exp(-0.5 * ((sel - center) / max(half / 2, 1.0)) ** 2)
            ww = ww * (0.2 + disp[sel])
            ww = ww / (ww.sum() + 1e-12)
            # weighted Kabsch
            mu0s = (ww[:, None] * ca0[sel]).sum(0)
            mu1s = (ww[:, None] * ca1[sel]).sum(0)
            X = (ca0[sel] - mu0s) * ww[:, None]
            Y = ca1[sel] - mu1s
            H = X.T @ Y
            try:
                U, _S, Vt = np.linalg.svd(H)
                d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
                R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
            except np.linalg.LinAlgError:
                R = np.eye(3)
        else:
            R = _kabsch_R(ca0[sel], ca1[sel])
            mu0s, mu1s = ca0[sel].mean(0), ca1[sel].mean(0)
        om = _so3_log(R)
        res = ca1[i] - ((ca0[i] - mu0s) @ R.T + mu1s)
        for ti, s in enumerate(fracs):
            t = _time(s, ease, herm)
            Rs = _so3_exp(om * t)
            mus = (1.0 - t) * mu0s + t * mu1s
            loc[ti, i] = (ca0[i] - mu0s) @ Rs.T + mus + t * res

    # multi-hinge domains by displacement gradient peaks
    hinge_path = screw.copy()
    if idx.size >= 12 and nhinge >= 1:
        d = disp.copy()
        grad = np.zeros(L)
        for a, b in zip(idx[:-1], idx[1:]):
            if b == a + 1:
                grad[b] = abs(d[b] - d[a])
        # pick top nhinge hinge residues along chain
        cand = idx[3:-3]
        order = cand[np.argsort(-grad[cand])]
        hinges = []
        for h in order:
            if all(abs(int(h) - int(hh)) > max(5, L // (nhinge * 4)) for hh in hinges):
                hinges.append(int(h))
            if len(hinges) >= nhinge:
                break
        hinges = sorted(hinges)
        bounds = [int(idx[0])] + hinges + [int(idx[-1]) + 1]
        domains = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            sel = np.arange(a, b)
            sel = sel[m[sel]]
            if sel.size < 3:
                continue
            R = _kabsch_R(ca0[sel], ca1[sel])
            mu0s, mu1s = ca0[sel].mean(0), ca1[sel].mean(0)
            om = _so3_log(R)
            res = ca1[sel] - ((ca0[sel] - mu0s) @ R.T + mu1s)
            domains.append((sel, mu0s, mu1s, om, res))
        for i, s in enumerate(fracs):
            t = _time(s, ease, herm)
            frame = lin[i].copy()
            for sel, mu0s, mu1s, om, res in domains:
                Rs = _so3_exp(om * t)
                mus = (1.0 - t) * mu0s + t * mu1s
                frame[sel] = (ca0[sel] - mu0s) @ Rs.T + mus + t * res
            hinge_path[i] = frame

    # residual PCA modes (top nmodes of endpoint displacement on masked residues)
    mode_path = lin.copy()
    if nmodes > 0 and m.sum() >= 6:
        idxm = np.where(m)[0]
        D = (ca1[idxm] - ca0[idxm])  # (n,3)
        # SVD of displacement as n x 3
        try:
            U, S, Vt = np.linalg.svd(D, full_matrices=False)
        except np.linalg.LinAlgError:
            U, S, Vt = None, None, None
        if U is not None:
            k = min(nmodes, U.shape[1], 3)
            recon = np.zeros_like(D)
            for j in range(k):
                recon += np.outer(U[:, j] * S[j], Vt[j])
            miss = D - recon
            beta = float(p.get("mode_mid_boost", 0.0))
            for i, s in enumerate(fracs):
                t = _time(s, ease, herm)
                mid = beta * np.sin(np.pi * s)
                delta = np.zeros((L, 3))
                delta[idxm] = (t + mid) * recon + t * miss
                mode_path[i] = ca0 + delta

    # blend
    ws = np.array([
        float(p.get("w_lin", 0.1)),
        float(p.get("w_screw", 0.5)),
        float(p.get("w_multi", 0.0)),
        float(p.get("w_local", 0.2)),
        float(p.get("w_hinge", 0.0)),
        float(p.get("w_mode", 0.0)),
    ], dtype=np.float64)
    ws = np.maximum(ws, 0.0)
    if ws.sum() < 1e-12:
        ws[0] = 1.0
    ws = ws / ws.sum()
    parts = [lin, screw, multi, loc, hinge_path, mode_path]
    out = sum(w * x for w, x in zip(ws, parts))

    # gentle bond restore
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
    # inject PARAMS into PROG
    src = PROG.replace("PARAMS = {}", "PARAMS = " + repr(params), 1)
    path.write_text(src, encoding="utf-8")


def eval_file(path: Path, n_windows=None, split="evolve") -> dict:
    return ev._score(str(path), n_windows=n_windows, split=split, timeout=180)


def balanced(es: float, hs: float) -> float:
    # reward joint improvement; soft-penalize holdout regression
    return 0.5 * es + 0.5 * hs - 0.35 * max(0.0, -hs)


def clip(x, a, b):
    return a if x < a else b if x > b else x


def sample(rng: random.Random) -> dict:
    raw = [rng.random() for _ in range(6)]
    s = sum(raw) + 1e-12
    w = [x / s for x in raw]
    return {
        "nseg": rng.choice([1, 2, 3, 4, 5]),
        "n_hinge": rng.choice([1, 1, 2, 3]),
        "n_modes": rng.choice([0, 1, 1, 2]),
        "ease": rng.uniform(0.0, 0.6),
        "hermite": rng.choice([0.0, 0.0, 0.2, 0.5, 0.8, 1.0]),
        "w_lin": w[0],
        "w_screw": w[1],
        "w_multi": w[2],
        "w_local": w[3],
        "w_hinge": w[4],
        "w_mode": w[5],
        "bond_strength": rng.choice([0.0, 0.0, 0.0, 0.05, 0.1]),
        "local_half": rng.randint(6, 16),
        "soft_local": rng.choice([0.0, 0.0, 0.5, 1.0]),
        "res_mid_boost": rng.choice([0.0, 0.0, 0.05, 0.1, 0.2]),
        "mode_mid_boost": rng.choice([0.0, 0.0, 0.05, 0.1]),
    }


def mutate(p: dict, rng: random.Random, scale: float = 0.35) -> dict:
    q = copy.deepcopy(p)
    keys = list(q.keys())
    for k in rng.sample(keys, k=rng.randint(1, 4)):
        if k in ("nseg", "n_hinge", "n_modes", "local_half"):
            step = rng.choice([-2, -1, 1, 2])
            lo, hi = {"nseg": (1, 6), "n_hinge": (1, 4), "n_modes": (0, 3), "local_half": (4, 18)}[k]
            q[k] = int(clip(q[k] + step, lo, hi))
        elif k.startswith("w_"):
            q[k] = float(max(0.0, q[k] + rng.gauss(0, 0.08 * scale)))
        elif k == "soft_local":
            q[k] = float(clip(q[k] + rng.gauss(0, 0.2 * scale), 0.0, 1.0))
        else:
            q[k] = float(clip(q[k] + rng.gauss(0, 0.08 * scale), 0.0, 1.0))
    ws = ["w_lin", "w_screw", "w_multi", "w_local", "w_hinge", "w_mode"]
    s = sum(max(0.0, q[k]) for k in ws) + 1e-12
    for k in ws:
        q[k] = max(0.0, q[k]) / s
    return q


# v2 champion + variants
SEEDS = [
    {"nseg": 1, "n_hinge": 1, "n_modes": 0, "ease": 0, "hermite": 0,
     "w_lin": 1, "w_screw": 0, "w_multi": 0, "w_local": 0, "w_hinge": 0, "w_mode": 0,
     "bond_strength": 0, "local_half": 13, "soft_local": 0, "res_mid_boost": 0, "mode_mid_boost": 0},
    {"nseg": 1, "n_hinge": 1, "n_modes": 0, "ease": 0, "hermite": 1,
     "w_lin": 1, "w_screw": 0, "w_multi": 0, "w_local": 0, "w_hinge": 0, "w_mode": 0,
     "bond_strength": 0, "local_half": 13, "soft_local": 0, "res_mid_boost": 0, "mode_mid_boost": 0},
    # v2 best
    {"nseg": 3, "n_hinge": 1, "n_modes": 1, "ease": 0, "hermite": 0,
     "w_lin": 0.085, "w_screw": 0.691, "w_multi": 0, "w_local": 0.200, "w_hinge": 0, "w_mode": 0.025,
     "bond_strength": 0, "local_half": 13, "soft_local": 0, "res_mid_boost": 0, "mode_mid_boost": 0.104},
    # more local / soft
    {"nseg": 3, "n_hinge": 2, "n_modes": 1, "ease": 0, "hermite": 0.15,
     "w_lin": 0.05, "w_screw": 0.45, "w_multi": 0.05, "w_local": 0.35, "w_hinge": 0.05, "w_mode": 0.05,
     "bond_strength": 0, "local_half": 11, "soft_local": 1.0, "res_mid_boost": 0.05, "mode_mid_boost": 0.05},
    # multi-hinge heavy
    {"nseg": 4, "n_hinge": 3, "n_modes": 2, "ease": 0.1, "hermite": 0.2,
     "w_lin": 0.1, "w_screw": 0.25, "w_multi": 0.15, "w_local": 0.2, "w_hinge": 0.25, "w_mode": 0.05,
     "bond_strength": 0, "local_half": 10, "soft_local": 0.5, "res_mid_boost": 0.1, "mode_mid_boost": 0.05},
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=600)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--screen-windows", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.environ["HUMAN_EVOLVE_DEVICE"] = args.device

    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "param_evolve_v3_log.jsonl"
    rng = random.Random(args.seed)
    best = None
    t0 = time.perf_counter()

    def consider(params, it, force_full=False, screen_sc=None):
        nonlocal best
        materialize(params, TRIAL)
        if screen_sc is None:
            scr = eval_file_score(TRIAL, args.screen_windows, "evolve")
            screen_sc = float(scr.get("combined_score", -1))
        do_full = force_full or best is None or screen_sc > best.get("screen", -9) - 0.003
        if not do_full:
            return screen_sc, None
        full_e = eval_file_score(TRIAL, None, "evolve")
        full_h = eval_file_score(TRIAL, None, "holdout")
        es = float(full_e.get("combined_score", -1))
        hs = float(full_h.get("combined_score", -1))
        bal = balanced(es, hs)
        rec = {
            "params": copy.deepcopy(params),
            "screen": screen_sc,
            "evolve": es,
            "holdout": hs,
            "balanced": bal,
            "metrics_e": full_e,
            "metrics_h": full_h,
            "iter": it,
        }
        if best is None or bal > best["balanced"] + 1e-9:
            best = rec
            print(
                f"[{it:4d}] NEW BEST bal={bal:+.5f}  e={es:+.5f} ({full_e.get('model_rmsd'):.4f}A)  "
                f"h={hs:+.5f} ({full_h.get('model_rmsd'):.4f}A)  "
                f"screw={params.get('w_screw'):.2f} local={params.get('w_local'):.2f} "
                f"hinge={params.get('w_hinge'):.2f} soft={params.get('soft_local'):.1f}"
            )
        return screen_sc, rec

    def eval_file_score(path, n_windows, split):
        return ev._score(str(path), n_windows=n_windows, split=split, timeout=180)

    print(f"=== param evolve v3: {args.iterations} iters device={args.device} ===")
    with open(log_path, "w", encoding="utf-8") as logf:
        for i, seed in enumerate(SEEDS):
            sc, rec = consider(seed, it=-(i + 1), force_full=True)
            logf.write(json.dumps({"iter": -(i + 1), "seed": True, "screen": sc,
                                   "evolve": None if not rec else rec["evolve"],
                                   "holdout": None if not rec else rec["holdout"],
                                   "balanced": None if not rec else rec["balanced"],
                                   "params": seed}) + "\n")

        for it in range(1, args.iterations + 1):
            scale = 1.0 if it < args.iterations // 3 else (0.5 if it < 2 * args.iterations // 3 else 0.25)
            if rng.random() < 0.2:
                cand = sample(rng)
            else:
                parent = best["params"] if best else rng.choice(SEEDS)
                cand = mutate(parent, rng, scale=scale)

            materialize(cand, TRIAL)
            scr = eval_file_score(TRIAL, args.screen_windows, "evolve")
            screen_sc = float(scr.get("combined_score", -1))
            _, rec = consider(cand, it=it, force_full=False, screen_sc=screen_sc)
            row = {
                "iter": it,
                "screen": screen_sc,
                "screen_rmsd": scr.get("model_rmsd"),
                "evolve": None if rec is None else rec["evolve"],
                "holdout": None if rec is None else rec["holdout"],
                "balanced": None if rec is None else rec["balanced"],
                "params": cand,
            }
            logf.write(json.dumps(row) + "\n")
            logf.flush()

            if it % 50 == 0:
                elapsed = time.perf_counter() - t0
                b = best
                print(
                    f"[{it:4d}/{args.iterations}] screen={screen_sc:+.4f}  "
                    f"best_bal={b['balanced']:+.5f} e={b['evolve']:+.5f} h={b['holdout']:+.5f}  "
                    f"{elapsed:.0f}s ({elapsed/it:.2f}s/it)"
                )

    materialize(best["params"], TRIAL)
    best_dir = OUT / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TRIAL, best_dir / "best_program_v3.py")
    shutil.copy2(TRIAL, best_dir / "best_program.py")
    info = {
        "source": "param_evolve_v3",
        "iterations": args.iterations,
        "device": args.device,
        "best": {
            "iter": best["iter"],
            "balanced": best["balanced"],
            "evolve": best["evolve"],
            "holdout": best["holdout"],
            "params": best["params"],
            "metrics_evolve": best["metrics_e"],
            "metrics_holdout": best["metrics_h"],
        },
    }
    (best_dir / "best_program_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (OUT / "param_evolve_v3_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("\n=== DONE v3 ===")
    print(json.dumps(info["best"], indent=2, default=str)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
