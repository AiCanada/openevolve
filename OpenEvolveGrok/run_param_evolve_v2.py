#!/usr/bin/env python
"""500–1000 iter param evolution v2: hermite + multi-blend, BALANCED evolve/holdout fitness.

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

SRC_V2 = EX / "param_program_v2.py"
TRIAL = EX / "_param_trial_v2.py"
OUT = EX / "openevolve_output_grok"


def clip(x, a, b):
    return a if x < a else b if x > b else x


def sample(rng: random.Random) -> dict:
    # Dirichlet-ish weights
    raw = [rng.random() for _ in range(6)]
    s = sum(raw) + 1e-12
    w = [x / s for x in raw]
    return {
        "nseg": rng.choice([1, 2, 3, 4, 5]),
        "ease": rng.uniform(0.0, 1.0),
        "hermite": rng.choice([0.0, 0.0, 0.25, 0.5, 0.75, 1.0]),
        "w_lin": w[0],
        "w_screw": w[1],
        "w_multi": w[2],
        "w_local": w[3],
        "w_hinge": w[4],
        "w_mode": w[5],
        "bond_strength": rng.choice([0.0, 0.0, 0.0, 0.1, 0.2]),
        "local_half": rng.randint(4, 14),
        "hinge_smooth": rng.randint(0, 4),
        "res_mid_boost": rng.choice([0.0, 0.0, 0.1, 0.2, 0.35]),
        "mode_mid_boost": rng.choice([0.0, 0.0, 0.1, 0.15]),
    }


def mutate(p: dict, rng: random.Random, scale: float = 1.0) -> dict:
    q = copy.deepcopy(p)
    keys = list(q.keys())
    for k in rng.sample(keys, k=rng.randint(1, 4)):
        if k == "nseg":
            q[k] = int(clip(q[k] + rng.choice([-1, 1, 2, -2]), 1, 6))
        elif k == "local_half":
            q[k] = int(clip(q[k] + rng.randint(-3, 3), 3, 16))
        elif k == "hinge_smooth":
            q[k] = int(clip(q[k] + rng.randint(-2, 2), 0, 6))
        elif k.startswith("w_"):
            q[k] = float(max(0.0, q[k] + rng.gauss(0, 0.12 * scale)))
        else:
            q[k] = float(clip(q[k] + rng.gauss(0, 0.12 * scale), 0.0, 1.0))
    # renormalize blend weights
    ws = ["w_lin", "w_screw", "w_multi", "w_local", "w_hinge", "w_mode"]
    s = sum(max(0.0, q[k]) for k in ws) + 1e-12
    for k in ws:
        q[k] = max(0.0, q[k]) / s
    return q


SEEDS = [
    # pure linear
    {"nseg": 1, "ease": 0, "hermite": 0, "w_lin": 1, "w_screw": 0, "w_multi": 0,
     "w_local": 0, "w_hinge": 0, "w_mode": 0, "bond_strength": 0, "local_half": 8,
     "hinge_smooth": 2, "res_mid_boost": 0, "mode_mid_boost": 0},
    # pure hermite
    {"nseg": 1, "ease": 0, "hermite": 1, "w_lin": 1, "w_screw": 0, "w_multi": 0,
     "w_local": 0, "w_hinge": 0, "w_mode": 0, "bond_strength": 0, "local_half": 8,
     "hinge_smooth": 2, "res_mid_boost": 0, "mode_mid_boost": 0},
    # previous 500 winner
    {"nseg": 3, "ease": 0, "hermite": 0, "w_lin": 0.214, "w_screw": 0.505, "w_multi": 0,
     "w_local": 0.259, "w_hinge": 0.15, "w_mode": 0, "bond_strength": 0, "local_half": 11,
     "hinge_smooth": 0, "res_mid_boost": 0, "mode_mid_boost": 0},
    # screw heavy
    {"nseg": 3, "ease": 0.2, "hermite": 0.3, "w_lin": 0.2, "w_screw": 0.5, "w_multi": 0.15,
     "w_local": 0.1, "w_hinge": 0.05, "w_mode": 0, "bond_strength": 0, "local_half": 10,
     "hinge_smooth": 1, "res_mid_boost": 0.15, "mode_mid_boost": 0},
    # hermite + light screw
    {"nseg": 2, "ease": 0, "hermite": 0.7, "w_lin": 0.55, "w_screw": 0.3, "w_multi": 0.1,
     "w_local": 0.05, "w_hinge": 0, "w_mode": 0, "bond_strength": 0, "local_half": 8,
     "hinge_smooth": 2, "res_mid_boost": 0.1, "mode_mid_boost": 0.05},
]


def materialize(params: dict, path: Path) -> None:
    src = SRC_V2.read_text(encoding="utf-8")
    body = "PARAMS = " + repr(params)
    src2 = re.sub(r"PARAMS = \{[\s\S]*?\n\}", body, src, count=1)
    path.write_text(src2, encoding="utf-8")


def eval_path(path: Path, n_windows=None, split="evolve") -> dict:
    return ev._score(str(path), n_windows=n_windows, split=split, timeout=180)


def balanced(e_score: float, h_score: float) -> float:
    """Prefer improving both; punish holdout regressions."""
    return 0.55 * e_score + 0.45 * h_score - 0.5 * max(0.0, -h_score)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--screen-windows", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.environ["HUMAN_EVOLVE_DEVICE"] = args.device

    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "param_evolve_v2_log.jsonl"
    rng = random.Random(args.seed)

    # evaluate seeds fully
    best = None
    t0 = time.perf_counter()
    history = []

    def consider(params, it, force_full=False, screen_sc=None):
        nonlocal best
        materialize(params, TRIAL)
        if screen_sc is None:
            scr = eval_path(TRIAL, args.screen_windows, "evolve")
            screen_sc = float(scr.get("combined_score", -1))
        else:
            scr = {"combined_score": screen_sc}

        # full if promising or forced
        do_full = force_full or best is None or screen_sc > best.get("screen", -9) - 0.005
        if not do_full:
            return screen_sc, None

        full_e = eval_path(TRIAL, None, "evolve")
        full_h = eval_path(TRIAL, None, "holdout")
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
        if best is None or bal > best["balanced"]:
            best = rec
            print(
                f"[{it:4d}] NEW BEST bal={bal:+.5f}  evolve={es:+.5f} ({full_e.get('model_rmsd'):.4f}A)  "
                f"holdout={hs:+.5f} ({full_h.get('model_rmsd'):.4f}A)  herm={params.get('hermite'):.2f} "
                f"w=[{params.get('w_lin'):.2f},{params.get('w_screw'):.2f},{params.get('w_local'):.2f}]"
            )
        return screen_sc, rec

    print(f"=== param evolve v2: {args.iterations} iters device={args.device} ===")
    with open(log_path, "w", encoding="utf-8") as logf:
        # seeds
        for i, seed in enumerate(SEEDS):
            sc, rec = consider(seed, it=-(i + 1), force_full=True)
            logf.write(json.dumps({"iter": -(i + 1), "seed": True, "screen": sc,
                                   "evolve": None if not rec else rec["evolve"],
                                   "holdout": None if not rec else rec["holdout"],
                                   "balanced": None if not rec else rec["balanced"],
                                   "params": seed}) + "\n")

        for it in range(1, args.iterations + 1):
            scale = 1.0 if it < args.iterations // 2 else 0.45
            if rng.random() < 0.25:
                cand = sample(rng)
            else:
                parent = best["params"] if best else rng.choice(SEEDS)
                cand = mutate(parent, rng, scale=scale)

            materialize(cand, TRIAL)
            scr = eval_path(TRIAL, args.screen_windows, "evolve")
            screen_sc = float(scr.get("combined_score", -1))
            sc, rec = consider(cand, it=it, force_full=False, screen_sc=screen_sc)

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
            history.append(row)

            if it % 50 == 0:
                elapsed = time.perf_counter() - t0
                b = best
                print(
                    f"[{it:4d}/{args.iterations}] screen={screen_sc:+.4f}  "
                    f"best_bal={b['balanced']:+.5f} e={b['evolve']:+.5f} h={b['holdout']:+.5f}  "
                    f"{elapsed:.0f}s ({elapsed/it:.2f}s/it)"
                )

    # write best
    materialize(best["params"], TRIAL)
    best_dir = OUT / "best"
    best_dir.mkdir(exist_ok=True)
    shutil.copy2(TRIAL, best_dir / "best_program_v2.py")
    # also promote if better balanced than previous
    shutil.copy2(TRIAL, best_dir / "best_program.py")
    info = {
        "source": "param_evolve_v2",
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
    (OUT / "param_evolve_v2_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("\n=== DONE v2 ===")
    print(json.dumps(info["best"], indent=2, default=str)[:2000])
    print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
