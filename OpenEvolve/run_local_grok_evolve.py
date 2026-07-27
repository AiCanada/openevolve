#!/usr/bin/env python
"""Local Grok evolution loop (no Claude, no XAI_API_KEY required).

This session's algorithms are generated as candidate files; this script scores
them with the HUMAN GPU evaluator and keeps the Pareto / best winner.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EX = ROOT / "examples" / "human_interp_path"
CAND = EX / "candidates"
OUT = EX / "openevolve_output_grok"
EVAL = EX / "evaluator.py"


def score(program: Path, split: str = "evolve") -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env.setdefault("HUMAN_EVOLVE_DEVICE", "cuda")
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, str(EVAL), str(program), split],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        return {
            "combined_score": -1.0,
            "error": (r.stderr or r.stdout)[-500:],
            "eval_seconds": dt,
            "valid": 0.0,
        }
    # stdout may have noise; find last JSON object
    text = r.stdout.strip()
    try:
        # last {...}
        i = text.rfind("{")
        data = json.loads(text[i:])
    except Exception as e:
        return {"combined_score": -1.0, "error": f"parse: {e} raw={text[-300:]}", "eval_seconds": dt}
    data["eval_seconds"] = round(dt, 4)
    return data


def main():
    os.environ.setdefault("HUMAN_EVOLVE_DEVICE", "cuda")
    # generate candidates
    subprocess.check_call([sys.executable, str(CAND / "gen_candidates.py")], cwd=str(ROOT))
    programs = sorted(CAND.glob("*.py"))
    programs = [p for p in programs if p.name != "gen_candidates.py"]
    print(f"Scoring {len(programs)} candidates on device={os.environ.get('HUMAN_EVOLVE_DEVICE')}")

    rows = []
    for p in programs:
        print(f"--- {p.name} ---")
        ev = score(p, "evolve")
        ho = score(p, "holdout")
        row = {
            "name": p.name,
            "path": str(p),
            "evolve": ev,
            "holdout": ho,
            "score_evolve": float(ev.get("combined_score", -1)),
            "score_holdout": float(ho.get("combined_score", -1)),
        }
        rows.append(row)
        print(
            f"  evolve:  score={row['score_evolve']:+.4f}  rmsd={ev.get('model_rmsd')}  "
            f"base={ev.get('baseline_rmsd')}  win={ev.get('win_rate')}  dev={ev.get('device')}"
        )
        print(
            f"  holdout: score={row['score_holdout']:+.4f}  rmsd={ho.get('model_rmsd')}  "
            f"win={ho.get('win_rate')}"
        )

    # rank by balanced score (min of evolve/holdout) then mean — avoid overfit
    for r in rows:
        r["score_min"] = min(r["score_evolve"], r["score_holdout"])
        r["score_mean"] = 0.5 * (r["score_evolve"] + r["score_holdout"])
    rows.sort(key=lambda r: (r["score_min"], r["score_mean"], r["score_holdout"]), reverse=True)
    best = rows[0]

    OUT.mkdir(parents=True, exist_ok=True)
    best_dir = OUT / "best"
    best_dir.mkdir(exist_ok=True)
    shutil.copy2(best["path"], best_dir / "best_program.py")
    info = {
        "source": "local_grok_evolve",
        "best_name": best["name"],
        "metrics_evolve": best["evolve"],
        "metrics_holdout": best["holdout"],
        "ranking": [
            {
                "name": r["name"],
                "score_evolve": r["score_evolve"],
                "score_holdout": r["score_holdout"],
                "model_rmsd_evolve": r["evolve"].get("model_rmsd"),
                "model_rmsd_holdout": r["holdout"].get("model_rmsd"),
            }
            for r in rows
        ],
    }
    (best_dir / "best_program_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (OUT / "leaderboard.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print("\n=== LEADERBOARD (by holdout score) ===")
    for i, r in enumerate(rows, 1):
        print(
            f"{i:2d}. {r['name']:28s}  holdout={r['score_holdout']:+.4f}  "
            f"evolve={r['score_evolve']:+.4f}  "
            f"rmsd_e={r['evolve'].get('model_rmsd')}  rmsd_h={r['holdout'].get('model_rmsd')}"
        )
    print(f"\nBest -> {best_dir / 'best_program.py'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
