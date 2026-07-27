"""Transition-path interpolation for protein backbones (Project H.U.M.A.N.)."""
import numpy as np

# EVOLVE-BLOCK-START


def predict_interior(ca_start, ca_end, fracs, mask):
    ca0 = np.asarray(ca_start, dtype=np.float64)
    ca1 = np.asarray(ca_end, dtype=np.float64)
    out = np.empty((len(fracs), ca0.shape[0], 3), dtype=np.float64)
    for i, s in enumerate(fracs):
        out[i] = (1.0 - s) * ca0 + s * ca1
    return out

# EVOLVE-BLOCK-END


def run_interpolation(ca_start, ca_end, fracs, mask):
    return predict_interior(ca_start, ca_end, fracs, mask)
