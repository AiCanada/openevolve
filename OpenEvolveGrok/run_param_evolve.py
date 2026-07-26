#!/usr/bin/env python
"""500+ iteration parameter evolution for path interpolator (GPU scorer).

No Claude. No XAI_API_KEY required. Uses random + local search over PARAMS
in param_program.py. Fast screen on 25 windows; full 90-window confirm for elites.
"""
from __future__ import annotations

import copy
import json
import os
import random
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

# Import evaluator helpers directly for speed (no subprocess)
import importlib.util

spec = importlib.util.spec_from_file_location("param_program", EX / "param_program.py")
prog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prog)

spec_e = importlib.util.spec_from_file_location("evaluator", EX / "evaluator.py")
ev = importlib.util.module_from_spec(spec_e)
spec_e.loader.exec_module(ev)


def sample_params(rng: random.Random) -> dict:
    return {
        "nseg": rng.choice([1, 2, 3, 4, 5, 6]),
        "ease": rng.uniform(0.0, 1.0),
        "w_lin": rng.uniform(0.0, 1.0),
        "w_screw": rng.uniform(0.0, 1.0),
        "w_multi": rng.uniform(0.0, 1.0),
        "bond_strength": rng.choice([0.0, 0.0, 0.0, 0.1, 0.2, 0.35, 0.5]),
        "local_half": rng.randint(3, 12),
        "use_local": rng.choice([0.0, 0.0, 0.0, 0.15, 0.3, 0.5]),
        "hinge_smooth": rng.randint(1, 5),
        "use_hinge": rng.choice([0.0, 0.0, 0.0, 0.15, 0.3, 0.5]),
    }


def mutate(p: dict, rng: random.Random, scale: float = 1.0) -> dict:
    q = copy.deepcopy(p)
    keys = list(q.keys())
    # mutate 1-3 params
    for k in rng.sample(keys, k=rng.randint(1, 3)):
        if k == "nseg":
            q[k] = int(np_clip(q[k] + rng.choice([-2, -1, 1, 2]), 1, 8))
        elif k == "local_half":
            q[k] = int(np_clip(q[k] + rng.randint(-3, 3), 2, 16))
        elif k == "hinge_smooth":
            q[k] = int(np_clip(q[k] + rng.randint(-2, 2), 0, 8))
        elif k in ("use_local", "use_hinge", "ease", "bond_strength"):
            q[k] = float(np_clip(q[k] + rng.gauss(0, 0.15 * scale), 0.0, 1.0))
        else:
            q[k] = float(max(0.0, q[k] + rng.gauss(0, 0.2 * scale)))
    return q


def np_clip(x, a, b):
    return a if x < a else b if x > b else x


def score_params(params: dict, n_windows: int | None, split: str = "evolve") -> dict:
    prog.inject_params(params)
    # write temp program path that evaluator can load — use param_program itself
    # evaluator loads run_interpolation from file; inject_params already set in memory
    # but evaluator re-imports file fresh — so write a materialization
    return None  # placeholder — use file path approach


def materialize(params: dict, path: Path) -> None:
    src = (EX / "param_program.py").read_text(encoding="utf-8")
    # replace PARAMS dict body
    import re

    body = "PARAMS = " + repr(params) + "\n"
    src2 = re.sub(
        r"PARAMS = \{[\s\S]*?\n\}",
        body.rstrip(),
        src,
        count=1,
    )
    path.write_text(src2, encoding="utf-8")


def eval_file(path: Path, n_windows: int | None, split: str = "evolve") -> dict:
    return ev._score(str(path), n_windows=n_windows, split=split, timeout=120)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--screen-windows", type=int, default=25)
    ap.add_argument("--full-every", type=int, default=25)
    ap.add_argument("--device", default=os.environ.get("HUMAN_EVOLVE_DEVICE", "cuda"))
    args = ap.parse_args()

    os.environ["HUMAN_EVOLVE_DEVICE"] = args.device
    rng = random.Random(args.seed)
    out_dir = EX / "openevolve_output_grok"
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_path = EX / "_param_trial.py"
    log_path = out_dir / "param_evolve_log.jsonl"

    # baseline linear via empty multi weight? Use pure linear params
    linear = {
        "nseg": 1, "ease": 0.0, "w_lin": 1.0, "w_screw": 0.0, "w_multi": 0.0,
        "bond_strength": 0.0, "local_half": 6, "use_local": 0.0,
        "hinge_smooth": 2, "use_hinge": 0.0,
    }
    materialize(linear, trial_path)
    base = eval_file(trial_path, None, "evolve")
    print(f"Linear baseline full evolve: score={base.get('combined_score')} rmsd={base.get('model_rmsd')} device={base.get('device')}")

    # seed population
    pop = [linear] + [sample_params(rng) for _ in range(7)]
    # known good seed from gen1
    pop.append({
        "nseg": 3, "ease": 0.0, "w_lin": 0.25, "w_screw": 0.35, "w_multi": 0.40,
        "bond_strength": 0.0, "local_half": 6, "use_local": 0.0,
        "hinge_smooth": 2, "use_hinge": 0.0,
    })
    # hermite-like: pure linear with ease on residual path via high ease + lin weight
    pop.append({
        "nseg": 1, "ease": 1.0, "w_lin": 1.0, "w_screw": 0.0, "w_multi": 0.0,
        "bond_strength": 0.0, "local_half": 6, "use_local": 0.0,
        "hinge_smooth": 2, "use_hinge": 0.0,
    })

    best = None
    best_full = None
    history = []
    t0 = time.perf_counter()

    with open(log_path, "w", encoding="utf-8") as logf:
        for it in range(1, args.iterations + 1):
            if it <= len(pop):
                cand = pop[it - 1]
            elif rng.random() < 0.35:
                cand = sample_params(rng)
            else:
                parent = best["params"] if best else rng.choice(pop)
                cand = mutate(parent, rng, scale=1.0 if it < args.iterations // 2 else 0.5)

            materialize(cand, trial_path)
            scr = eval_file(trial_path, args.screen_windows, "evolve")
            sc = float(scr.get("combined_score", -1.0))

            # occasional full eval or when beats best screen
            do_full = (
                it == 1
                or it % args.full_every == 0
                or (best is None)
                or (sc > best.get("screen_score", -9) + 1e-6)
            )
            full = None
            if do_full and sc > -0.5:
                full = eval_file(trial_path, None, "evolve")
                fsc = float(full.get("combined_score", -1.0))
                if best_full is None or fsc > best_full["score"]:
                    best_full = {
                        "score": fsc,
                        "params": copy.deepcopy(cand),
                        "metrics": full,
                        "iteration": it,
                    }
                    # also holdout
                    ho = eval_file(trial_path, None, "holdout")
                    best_full["holdout"] = ho
                    best_full["holdout_score"] = float(ho.get("combined_score", -1.0))
                    print(
                        f"[{it:4d}] NEW BEST full evolve={fsc:+.5f} "
                        f"rmsd={full.get('model_rmsd'):.4f} "
                        f"holdout={best_full['holdout_score']:+.5f} "
                        f"params={cand}"
                    )

            if best is None or sc > best["screen_score"]:
                best = {"screen_score": sc, "params": copy.deepcopy(cand), "metrics": scr, "iteration": it}

            row = {
                "iter": it,
                "screen_score": sc,
                "screen_rmsd": scr.get("model_rmsd"),
                "full_score": None if full is None else full.get("combined_score"),
                "params": cand,
            }
            logf.write(json.dumps(row) + "\n")
            logf.flush()
            history.append(row)

            if it % 50 == 0:
                elapsed = time.perf_counter() - t0
                bf = best_full["score"] if best_full else None
                print(
                    f"[{it:4d}/{args.iterations}] screen_best={best['screen_score']:+.5f} "
                    f"full_best={bf}  elapsed={elapsed:.0f}s  ~{elapsed/it:.2f}s/it"
                )

    # finalize best
    if best_full is None:
        materialize(best["params"], trial_path)
        full = eval_file(trial_path, None, "evolve")
        ho = eval_file(trial_path, None, "holdout")
        best_full = {
            "score": float(full.get("combined_score", -1)),
            "params": best["params"],
            "metrics": full,
            "holdout": ho,
            "holdout_score": float(ho.get("combined_score", -1)),
            "iteration": best["iteration"],
        }

    materialize(best_full["params"], trial_path)
    best_dir = out_dir / "best"
    best_dir.mkdir(exist_ok=True)
    shutil.copy2(trial_path, best_dir / "best_program.py")
    # also copy as initial_program upgrade candidate
    info = {
        "source": "param_evolve_500",
        "iterations": args.iterations,
        "device": args.device,
        "best_iteration": best_full.get("iteration"),
        "params": best_full["params"],
        "metrics_evolve": best_full["metrics"],
        "metrics_holdout": best_full.get("holdout"),
        "score_evolve": best_full["score"],
        "score_holdout": best_full.get("holdout_score"),
        "linear_baseline": base,
    }
    (best_dir / "best_program_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out_dir / "param_evolve_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print("\n=== DONE ===")
    print(json.dumps({
        "score_evolve": best_full["score"],
        "score_holdout": best_full.get("holdout_score"),
        "model_rmsd": best_full["metrics"].get("model_rmsd"),
        "baseline_rmsd": best_full["metrics"].get("baseline_rmsd"),
        "params": best_full["params"],
        "iteration": best_full.get("iteration"),
    }, indent=2))
    print(f"Log: {log_path}")
    print(f"Best: {best_dir / 'best_program.py'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
