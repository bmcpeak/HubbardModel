"""
momentum_range.py -- wide-window bilinear extension of the Hubbard bootstrap.

"Momentum space" done validly: a k-grid of windowed Fourier modes is a
unitary rotation of the position bilinears on the same window (identical
bound), and sharply truncated M(k) sums are INVALID (Dirichlet kernel is
not positive). The real content is window size, so this file adds the
1-RDM (P) and hole (Q) moment blocks over an R x R window of separations
on top of the patch-level bases. Single spin suffices: the opposite-spin
copy is a canon duplicate, and cross-spin entries are Sz-charged (zero).

New variables: bilinear patterns at separations up to R-1, constrained
only by these blocks -- plus whatever wards / H-linear forms they enable.
"""

import time
from fractions import Fraction

from temp import (            # <- adjust module name if needed
    log, set_patch, build_M_bases, moment_matrix, split_variables,
    prune_charged, scrub_forms, linear_H_constraints, ward_constraints,
    objective, filling_constraint, assemble_and_solve,
    c_op, d_op, DN, OX, W)

R = 6                                       # separation window (R x R)
assert (R - 1) + OX + 1 < W, "shrink R or grow W"
WIN = [(i, j) for i in range(R) for j in range(R)]

def build_range_blocks():
    bc = [c_op(i, j, DN) for (i, j) in WIN]     # <d_x c_y>: 1-RDM  (P)
    bd = [d_op(i, j, DN) for (i, j) in WIN]     # <c_x d_y>: holes  (Q)
    return [moment_matrix(bc), moment_matrix(bd)]

def solve(level=3, t=1, U=8, nu=Fraction(7, 8)):
    set_patch(3)
    t0 = time.time()
    mats = [moment_matrix(b, label=f"block {i}")
            for i, b in enumerate(build_M_bases(level))]
    mats += build_range_blocks()
    log(f"fill: {time.time()-t0:.1f}s, blocks {[len(m) for m in mats]}")

    neutral, charged = split_variables(mats)
    span = neutral | charged | {0}

    hcons, h_oos = linear_H_constraints(neutral, span, t, U)
    wards, dropped = ward_constraints(neutral)
    prune_charged(mats)
    hcons = scrub_forms(hcons)
    wards = scrub_forms(wards)
    log(f"variables: {len(neutral)}; H-linear {len(hcons)} kept "
        f"({h_oos} out-of-span); wards {len(wards)}")

    obj = objective(t, U)
    fill = filling_constraint(nu)
    var_index = {k: i for i, k in enumerate(sorted(neutral))}
    eqs = ([(fill[0], fill[1])] + [(w, 0) for w in wards]
           + [(h, 0) for h in hcons])
    return assemble_and_solve(mats, obj, eqs, var_index)

if __name__ == "__main__":
    log("=== U = 0 diagnostic: range content should approach -1.402 ===")
    lo0, _ = solve(level=2, U=0)
    log("\n=== U = 8, level 3, patch 3 + range-6 bilinears ===")
    lo, _ = solve(level=3, U=8)
    log(f"\nU=0 bound:  {lo0:.6f}   (free fermions: about -1.402; "
        f"was -2.000 without range)")
    log(f"U=8 bound:  {lo:.6f}   (was -0.938438 without range)")