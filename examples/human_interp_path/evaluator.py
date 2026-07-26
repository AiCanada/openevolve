"""Evaluator: multi-component path score vs straight-line baseline.

Decomposes mid-path error into:
  * translation — COM path error
  * rotation    — Kabsch geodesic angle after COM centering
  * folding     — Kabsch-aligned Cα RMSD (internal / fold residual)  [PRIMARY]

total_score = translation_imp + rotation_imp + folding_imp
combined_score = 0.15*trans + 0.15*rot + 0.70*fold  (folding-weighted)

Evolve split for search; holdout split for post-hoc generalization.
Cascade: stage1 = 25 windows, stage2 = all 90 windows.
"""

import ast
import concurrent.futures
import importlib.util
import math
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "windows.npz")
_CACHE = {}

# Prefer CUDA when available unless overridden: cpu|cuda|cuda:0
_DEVICE = os.environ.get("HUMAN_EVOLVE_DEVICE", "").strip().lower()


def _resolve_device():
    if _DEVICE in ("cpu", "none", "numpy"):
        return "cpu"
    try:
        import torch

        if _DEVICE.startswith("cuda") and torch.cuda.is_available():
            return _DEVICE if ":" in _DEVICE else "cuda"
        if not _DEVICE and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _data(split="evolve"):
    if split not in _CACHE:
        z = np.load(_DATA)
        _CACHE[split] = (z[f"{split}_ca"], z[f"{split}_mask"])
    return _CACHE[split]


# ---------------------------------------------------------------------------
# Integrity: the candidate gets ONLY the two endpoints. Reaching the cached ground
# truth — or the filesystem at all — is cheating, not an algorithm. Checked with the
# AST rather than substring matching, so words like "global" or "closed" in comments
# and docstrings cannot trip it.
# ---------------------------------------------------------------------------
_ALLOWED_MODULES = {
    "numpy", "math", "itertools", "functools", "collections", "typing",
    "warnings", "dataclasses", "operator", "scipy",
    # Optional: pure torch math (no data I/O). GPU kernels must not load windows.npz.
    "torch",
}
_BANNED_CALLS = {"open", "eval", "exec", "compile", "__import__", "input", "exit", "quit"}
_BANNED_LOADERS = {"load", "loadtxt", "genfromtxt", "fromfile", "load_npz", "memmap"}


def _integrity_check(program_path):
    """Return a reason string if the program is not a self-contained algorithm."""
    with open(program_path, "r", encoding="utf-8-sig") as fh:
        src = fh.read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return f"syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in _ALLOWED_MODULES:
                    return f"disallowed import: {a.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root and root not in _ALLOWED_MODULES:
                return f"disallowed import from: {node.module}"
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in _BANNED_CALLS:
                return f"disallowed call: {f.id}()"
            if isinstance(f, ast.Attribute) and f.attr in _BANNED_LOADERS:
                return f"disallowed data load: .{f.attr}()"
        elif isinstance(node, ast.Attribute):
            if node.attr in ("__globals__", "__code__", "__builtins__", "__file__"):
                return f"disallowed attribute: {node.attr}"
    return None


def _kabsch_rmsd_np(a, b):
    """RMSD after optimal rigid superposition of a onto b. a, b: (n,3)."""
    if a.shape[0] < 3:
        return float("nan")
    ac = a - a.mean(0)
    bc = b - b.mean(0)
    try:
        U, _S, Vt = np.linalg.svd(ac.T @ bc)
    except np.linalg.LinAlgError:
        return float("nan")
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt((((ac @ R.T) - bc) ** 2).sum(1).mean()))


def _frame_components(pred, true, mask):
    """Decompose frame error into translation, rotation, folding.

    * translation — COM distance (Å)
    * rotation    — Kabsch geodesic angle after COM centering (rad)
    * folding     — Kabsch-aligned Cα RMSD = internal / fold residual (Å)

    Matches HUMAN training intuition: total ≈ trans + rot + fold.
    """
    m = np.asarray(mask, dtype=bool)
    if m.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    p = np.asarray(pred, dtype=np.float64)[m]
    t = np.asarray(true, dtype=np.float64)[m]
    # COM error scaled by true radius of gyration so ~0.2 Å drift on a ~15–25 Å
    # chain is a small fractional penalty (local-screw folds often move COM slightly).
    rg = float(np.sqrt(((t - t.mean(0)) ** 2).sum(1).mean()))
    e_trans = float(np.linalg.norm(p.mean(0) - t.mean(0)) / max(rg, 1.0))
    pc, tc = p - p.mean(0), t - t.mean(0)
    try:
        U, _S, Vt = np.linalg.svd(pc.T @ tc)
    except np.linalg.LinAlgError:
        return e_trans, float("nan"), float("nan")
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    if d == 0:
        d = 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    cos_t = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    e_rot = float(np.arccos(cos_t))
    e_fold = float(np.sqrt((((pc @ R.T) - tc) ** 2).sum(1).mean()))
    return e_trans, e_rot, e_fold


# combined_score weights — folding is primary for the next evolution run.
# Translation is soft-weighted: local-screw folds often move COM slightly;
# a hard relative COM penalty would erase fold gains (see total_score for full sum).
_W_TRANS, _W_ROT, _W_FOLD = 0.05, 0.15, 0.80


def _kabsch_rmsd_torch_batch(pred, true, mask, device):
    """Batched Kabsch RMSD on GPU.

    pred, true: (N, L, 3)  mask: (N, L) bool
    Returns: (N,) float64 numpy RMSDs (nan if <3 valid atoms).
    """
    import torch

    p = torch.as_tensor(pred, device=device, dtype=torch.float64)
    t = torch.as_tensor(true, device=device, dtype=torch.float64)
    m = torch.as_tensor(mask, device=device, dtype=torch.float64)
    N, L, _ = p.shape
    n = m.sum(dim=1).clamp(min=1.0)
    # centre
    mu_p = (p * m[..., None]).sum(1) / n[:, None]
    mu_t = (t * m[..., None]).sum(1) / n[:, None]
    pc = (p - mu_p[:, None, :]) * m[..., None]
    tc = (t - mu_t[:, None, :]) * m[..., None]
    # H = pc^T tc  -> (N,3,3)
    H = torch.einsum("nli,nlj->nij", pc, tc)
    try:
        U, _S, Vh = torch.linalg.svd(H)
    except Exception:
        return np.full(N, np.nan)
    # det fix
    R0 = torch.matmul(Vh.transpose(-2, -1), U.transpose(-2, -1))
    det = torch.linalg.det(R0)
    D = torch.eye(3, device=device, dtype=torch.float64).expand(N, 3, 3).clone()
    D[:, 2, 2] = torch.sign(det).clamp(min=-1.0)  # if det==0, sign=0 -> set later
    D[:, 2, 2] = torch.where(D[:, 2, 2] == 0, torch.ones_like(D[:, 2, 2]), D[:, 2, 2])
    R = torch.matmul(Vh.transpose(-2, -1), torch.matmul(D, U.transpose(-2, -1)))
    aligned = torch.einsum("nij,nlj->nli", R, pc)
    sq = ((aligned - tc) ** 2).sum(-1)  # (N,L)
    # mean over valid only
    rmsd = torch.sqrt((sq * m).sum(1) / n)
    rmsd = torch.where(n >= 3, rmsd, torch.full_like(rmsd, float("nan")))
    return rmsd.detach().cpu().numpy()


def _score(program_path, n_windows=None, split="evolve", timeout=120):
    bad = _integrity_check(program_path)
    if bad:
        return {"combined_score": -1.0, "improvement_pct": -100.0, "win_rate": 0.0,
                "valid": 0.0, "error": bad, "device": "n/a"}

    device = _resolve_device()
    X, M = _data(split)
    n = X.shape[0] if n_windows is None else min(n_windows, X.shape[0])
    T = X.shape[1]
    fracs = np.arange(1, T - 1, dtype=np.float64) / float(T - 1)
    mid = (fracs >= 0.35) & (fracs <= 0.65)
    mid_idx = np.where(mid)[0]

    spec = importlib.util.spec_from_file_location("candidate", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run_interpolation"):
        return {"combined_score": -1.0, "improvement_pct": -100.0, "win_rate": 0.0,
                "valid": 0.0, "error": "no run_interpolation()", "device": device}

    preds = []
    trues = []
    bases = []
    masks = []
    failures, last_err = 0, ""

    for w in range(n):
        m = M[w]
        ca = X[w].astype(np.float64)
        ca0, ca1, true_int = ca[0], ca[-1], ca[1:-1]
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(mod.run_interpolation, ca0.copy(), ca1.copy(),
                                fracs.copy(), m.copy())
                pred = np.asarray(fut.result(timeout=timeout), dtype=np.float64)
        except Exception as e:
            failures += 1
            last_err = f"{type(e).__name__}: {e}"[:200]
            continue
        if pred.shape != true_int.shape or not np.isfinite(pred[:, m]).all():
            failures += 1
            last_err = f"bad output: shape {pred.shape} vs {true_int.shape} or non-finite"
            continue

        # mid-path only
        for k in mid_idx:
            base_k = (1.0 - fracs[k]) * ca0 + fracs[k] * ca1
            preds.append(pred[k])
            trues.append(true_int[k])
            bases.append(base_k)
            masks.append(m)

    if not preds:
        return {"combined_score": -1.0, "improvement_pct": -100.0, "win_rate": 0.0,
                "valid": 0.0, "error": last_err or "no scorable windows", "device": device}

    P = np.stack(preds, 0)
    Tgt = np.stack(trues, 0)
    B = np.stack(bases, 0)
    Mk = np.stack(masks, 0)

    # Per mid-frame components: (trans, rot, fold) for model and linear baseline
    n_mid = int(mid_idx.size)
    n_ok_windows = len(P) // max(n_mid, 1)
    win_t_m, win_r_m, win_f_m = [], [], []
    win_t_b, win_r_b, win_f_b = [], [], []
    for w in range(n_ok_windows):
        sl = slice(w * n_mid, (w + 1) * n_mid)
        tms, rms, fms = [], [], []
        tbs, rbs, fbs = [], [], []
        for k in range(sl.start, sl.stop):
            et, er, ef = _frame_components(P[k], Tgt[k], Mk[k])
            bt, br, bf = _frame_components(B[k], Tgt[k], Mk[k])
            if np.isfinite(et) and np.isfinite(bt):
                tms.append(et); tbs.append(bt)
            if np.isfinite(er) and np.isfinite(br):
                rms.append(er); rbs.append(br)
            if np.isfinite(ef) and np.isfinite(bf):
                fms.append(ef); fbs.append(bf)
        if tms:
            win_t_m.append(float(np.median(tms))); win_t_b.append(float(np.median(tbs)))
        if rms:
            win_r_m.append(float(np.median(rms))); win_r_b.append(float(np.median(rbs)))
        if fms:
            win_f_m.append(float(np.median(fms))); win_f_b.append(float(np.median(fbs)))

    if not win_f_m:
        return {"combined_score": -1.0, "improvement_pct": -100.0, "win_rate": 0.0,
                "valid": 0.0, "error": last_err or "no finite mid-path scores", "device": device}

    def _paired_imp(wm, wb, floor):
        """Median PER-WINDOW relative improvement (genuinely paired), clipped.

        Each window is compared against ITS OWN baseline, then the medians are taken.
        A ratio of two INDEPENDENTLY computed medians is not a paired comparison and
        is exploitable: a candidate can shift one distribution's median while losing
        most head-to-head windows. That is not hypothetical — the previous version of
        this function was gamed exactly so, yielding rotation_score +2.04 % at
        win_rate_rot 0.30 (i.e. worse on 70 % of windows), which supplied 61 % of that
        program's combined_score. Keep this paired.
        """
        wm, wb = np.asarray(wm, float), np.asarray(wb, float)
        if len(wm) == 0:
            return 0.0, float("nan"), float("nan"), 0.0, 0
        med_m, med_b = float(np.median(wm)), float(np.median(wb))
        denom = np.maximum(np.abs(wb), float(floor))
        imp = float(np.clip(np.median((wb - wm) / denom), -1.0, 2.0))
        wins = int((wm < wb).sum())
        return imp, med_m, med_b, float(wins / max(len(wm), 1)), wins

    t_imp, t_m, t_b, t_wr, _ = _paired_imp(win_t_m, win_t_b, floor=0.01)
    r_imp, r_m, r_b, r_wr, _ = _paired_imp(win_r_m, win_r_b, floor=0.01)
    f_imp, f_m, f_b, f_wr, f_wins = _paired_imp(win_f_m, win_f_b, floor=0.5)

    n_w = len(win_f_m)
    mu, sd = n_w / 2.0, math.sqrt(n_w) / 2.0
    z = (abs(f_wins - mu) - 0.5) / sd if sd > 0 else 0.0
    p_value = float(math.erfc(z / math.sqrt(2)))
    paired_pct = 100.0 * f_imp
    median_ratio_pct = 100.0 * (1.0 - f_m / f_b) if f_b > 0 else 0.0

    # total = translation + rotation + folding (full multi-objective sum)
    total = t_imp + r_imp + f_imp
    # combined_score: fold-weighted; soft-cap COM downside so fold gains surface
    t_soft = float(np.clip(t_imp, -0.15, 2.0))
    combined = _W_TRANS * t_soft + _W_ROT * r_imp + _W_FOLD * f_imp
    valid = 1.0 - failures / max(n, 1)

    out = {
        "combined_score": float(combined) * valid,
        "total_score": float(total) * valid,
        "translation_score": float(t_imp) * valid,
        "rotation_score": float(r_imp) * valid,
        "folding_score": float(f_imp) * valid,
        "paired_improvement_pct": float(paired_pct),
        "median_ratio_pct": float(median_ratio_pct),
        "improvement_pct": float(paired_pct),
        "win_rate": float(f_wr),
        "sign_test_p": p_value,
        "gate_pass": 1.0 if (paired_pct >= 10.0 and f_wins > mu and p_value < 0.05) else 0.0,
        "model_rmsd": float(f_m),
        "baseline_rmsd": float(f_b),
        "model_trans": float(t_m),
        "baseline_trans": float(t_b),
        "model_rot": float(r_m),
        "baseline_rot": float(r_b),
        "win_rate_trans": float(t_wr),
        "win_rate_rot": float(r_wr),
        "win_rate_fold": float(f_wr),
        "valid": float(valid),
        "n_windows": float(n_w),
        "device": str(device),
        "w_trans": float(_W_TRANS),
        "w_rot": float(_W_ROT),
        "w_fold": float(_W_FOLD),
    }
    if last_err:
        out["error"] = last_err
    return out


def evaluate_stage1(program_path):
    """Fast screen on 25 windows."""
    return _score(program_path, n_windows=25, split="evolve", timeout=30)


def evaluate_stage2(program_path):
    """Full evaluation on the 90-window evolve split."""
    return _score(program_path, n_windows=None, split="evolve", timeout=120)


def evaluate(program_path):
    """Non-cascade entry point (same as stage 2)."""
    return evaluate_stage2(program_path)


if __name__ == "__main__":
    import sys, json, time
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "initial_program.py")
    split = sys.argv[2] if len(sys.argv) > 2 else "evolve"
    t0 = time.perf_counter()
    res = _score(path, split=split, timeout=300)
    res["eval_seconds"] = round(time.perf_counter() - t0, 4)
    print(json.dumps(res, indent=2))
