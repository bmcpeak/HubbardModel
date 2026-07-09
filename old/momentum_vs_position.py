"""
momentum_vs_position.py -- P=3 level-2 bootstrap in position vs momentum basis.

The momentum basis here is the real windowed Fourier basis on the 3x3 patch:
f_a(x) in {1, cos(2pi x/3), sin(2pi x/3)} per axis, tensored -> 9 real modes,
an invertible linear map of the 9 position modes. Bilinear AND pair bases are
transformed. Since each momentum basis is an invertible recombination of the
corresponding position basis, the PSD constraints are equivalent and the two
bounds MUST agree to solver precision. Any disagreement = bug (most likely in
the recombination bookkeeping). The open question is speed: momentum entries
are dense linear forms (many patterns per entry), position entries are sparse;
same SDP, different nonzero structure.
"""

import time
import math
from fractions import Fraction
from itertools import combinations

from old.bootstrap_hubbard import (        # adjust module name if needed
    log, set_patch, moment_matrix, split_variables, prune_charged,
    ward_constraints, scrub_forms, objective, filling_constraint,
    assemble_and_solve, multiply, add, c_op, d_op, IDENTITY, DN, UP)

# ---- real Fourier profiles on 3 sites: exact rational-ish coefficients ----
# cos(2pi/3) = -1/2, sin(2pi/3) = sqrt(3)/2. We scale rows to clear
# denominators; scaling basis vectors changes nothing about PSD content.
S3 = math.sqrt(3)
PROFILES_1D = [
    (1.0, 1.0, 1.0),          # k = 0
    (2.0, -1.0, -1.0),        # cos, scaled by 2
    (0.0, S3, -S3),           # sin, scaled by 2
]

def smeared(op_builder, s, prof2d):
    """Sum_x prof2d[x] * op(x, s) over the 3x3 patch."""
    out = {}
    for i in range(3):
        for j in range(3):
            w = prof2d[i][j]
            if w != 0.0:
                out = add(out, op_builder(i, j, s), scale=w)
    return out

def fourier_modes(op_builder, s):
    return [smeared(op_builder, s,
                    [[PROFILES_1D[a][i] * PROFILES_1D[b][j]
                      for j in range(3)] for i in range(3)])
            for a in range(3) for b in range(3)]

def bases_position():
    P3 = [(i, j) for i in range(3) for j in range(3)]
    q0  = [IDENTITY] + [multiply(d_op(*p, a), c_op(*q, b))
                        for p in P3 for q in P3 for a in (DN, UP) for b in (DN, UP)]
    qm1 = [c_op(i, j, s) for (i, j) in P3 for s in (DN, UP)]
    qp1 = [d_op(i, j, s) for (i, j) in P3 for s in (DN, UP)]
    cs  = [c_op(i, j, s) for (i, j) in P3 for s in (DN, UP)]
    qm2 = [multiply(a, b) for a, b in combinations(cs, 2)]
    ds  = [d_op(i, j, s) for (i, j) in P3 for s in (DN, UP)]
    qp2 = [multiply(a, b) for a, b in combinations(ds, 2)]
    return [q0, qm1, qp1, qm2, qp2]

def bases_momentum():
    ck = fourier_modes(c_op, DN) + fourier_modes(c_op, UP)   # 18 modes
    dk = fourier_modes(d_op, DN) + fourier_modes(d_op, UP)
    q0  = [IDENTITY] + [multiply(a, b) for a in dk for b in ck]
    qm1 = ck
    qp1 = dk
    qm2 = [multiply(a, b) for a, b in combinations(ck, 2)]
    qp2 = [multiply(a, b) for a, b in combinations(dk, 2)]
    return [q0, qm1, qp1, qm2, qp2]

def run(bases, tag, t=1, U=8, nu=Fraction(7, 8)):
    t0 = time.time()
    mats = [moment_matrix(b) for b in bases]
    t_fill = time.time() - t0
    log(f"[{tag}] fill {t_fill:.1f}s, blocks {[len(m) for m in mats]}")

    neutral, charged = split_variables(mats)
    wards, _ = ward_constraints(neutral)
    prune_charged(mats)
    wards = scrub_forms(wards)

    nnz = sum(len(e) for m in mats for row in m for e in row)
    log(f"[{tag}] variables {len(neutral)}, wards {len(wards)}, "
        f"total entry nonzeros {nnz}")

    obj = objective(t, U)
    fill = filling_constraint(nu)
    var_index = {k: i for i, k in enumerate(sorted(neutral))}
    eqs = [(fill[0], fill[1])] + [(w, 0) for w in wards]

    t0 = time.time()
    lower, _ = assemble_and_solve(mats, obj, eqs, var_index)
    return lower, t_fill, time.time() - t0

if __name__ == "__main__":
    set_patch(3)
    lo_p, fp, sp = run(bases_position(), "position")
    lo_k, fk, sk = run(bases_momentum(), "momentum")
    log(f"\nposition: {lo_p:.6f}   fill {fp:.1f}s  solve {sp:.1f}s")
    log(f"momentum: {lo_k:.6f}   fill {fk:.1f}s  solve {sk:.1f}s")
    log(f"difference: {abs(lo_p - lo_k):.2e}  (must be ~solver tolerance)")