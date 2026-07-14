"""Driver: patch-basis Hubbard bootstrap.

    python run.py                 # 3x3, level 2, U=8, t=1, nu=7/8, cvxpy/SCS
    python run.py --mosek         # MOSEK backend (dual form)
    python run.py --cross-check   # solve with all backends and compare

Lifecycle (order matters):
    basis -> decompose (orbit path) -> audit -> assemble -> validate ->
    freeze table -> drop raw memo -> FREE-STATE FEASIBILITY CHECK -> solve.

The feasibility check is mandatory: the exact-filling Gaussian state's
moment vector must satisfy every constraint; a failure means a sign bug
somewhere upstream and the run aborts before wasting solver time.

The bound is on the convex envelope of e(nu) (Maxwell construction) —
the honest statement in a phase-separation regime. No Ward identities yet.
"""

from __future__ import annotations

import sys
import time

from modes import Lattice
from hamiltonian import Hubbard
from symmetry import PointGroup, decompose, audit_decomposition
from moments import MomentTable
from basis_patch import patch_basis, patch_center
from assemble import assemble_problem, validate_problem
from free_fermions import TorusGaussianState, WickEvaluator
import sdp


def run(N=3, L=2, t=1, U=8, filling=7 / 8, backend="cvxpy", verbose=True,
        solver_log=False, skip_feasibility=False):
    lat = Lattice(N)
    hub = Hubbard(lat, t=t, U=U)

    t0 = time.perf_counter()
    ops = patch_basis(lat, 0, 0, N, L)
    pg = PointGroup(lat, patch_center(0, 0, N), include_flip=True)
    blocks = decompose(pg, ops)
    t1 = time.perf_counter()
    if verbose:
        print(f"basis: {len(ops)} monomials -> {len(blocks)} blocks, "
              f"largest {max(len(b.ops) for b in blocks)}  ({t1 - t0:.1f}s)")
        audit_decomposition(pg, ops, blocks, verbose=True)

    table = MomentTable(lat)
    prob = assemble_problem(blocks, table, hub, (0, 0), filling,
                            progress=verbose)
    validate_problem(prob)
    table.freeze()
    t2 = time.perf_counter()
    if verbose:
        print(prob.summary())
        print(f"assemble+validate: {t2 - t1:.1f}s")

    # mandatory end-to-end sign check: exact-filling Gaussian state
    tor = TorusGaussianState(64, filling / 2, t=t)
    wick = WickEvaluator(lat, lambda a, b: tor.g(a[0] - b[0], a[1] - b[1]))
    e_var = 2 * tor.e_kin_per_site_per_spin + U * wick.double_occupancy()
    if not skip_feasibility:
        y_free = wick.moment_vector(table)
        diag = sdp.check_feasibility(prob, y_free, tol=1e-7)
        if verbose:
            print(f"free-state check: eq viol {diag['eq_violation']:.2e}, "
                  f"min block eig {diag['min_block_eig']:.2e} "
                  f"({diag['worst_block']})")
        if not diag["feasible"]:
            raise RuntimeError(f"FREE-STATE FEASIBILITY FAILURE — sign bug "
                               f"upstream, aborting: {diag}")
    dropped = table.drop_raw_memo()
    t3 = time.perf_counter()
    if verbose:
        print(f"feasibility check: {t3 - t2:.1f}s "
              f"(dropped {dropped} raw memo entries)")

    if backend == "none":
        print("(assemble-only: stopping before solve)")
        return None, prob, None
    if backend == "cross-check":
        sdp.cross_check(prob)
        return None, prob, None
    solve = sdp.solve_mosek if backend == "mosek" else sdp.solve_cvxpy
    bound, y, status = solve(prob, verbose=solver_log)
    t4 = time.perf_counter()
    if verbose:
        print(f"solve ({backend}): {t4 - t3:.1f}s, status {status}")
        print(f"\nlower bound  e(nu={filling}) >= {bound:.6f}")
        print(f"Gaussian variational reference (exact nu): {e_var:.6f}")
    return bound, prob, y


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Patch-basis Hubbard bootstrap")
    ap.add_argument("--patch", type=int, default=3, help="N for NxN patch")
    ap.add_argument("--level", type=int, default=2, help="max operator degree L")
    ap.add_argument("--U", type=float, default=8)
    ap.add_argument("--t", type=float, default=1)
    ap.add_argument("--nu", type=float, default=7/8)
    ap.add_argument("--mosek", action="store_true")
    ap.add_argument("--cross-check", action="store_true",
                    help="solve with all backends and compare")
    ap.add_argument("--assemble-only", action="store_true",
                    help="stop after assembly+validation+feasibility")
    ap.add_argument("--skip-feasibility", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="stream the solver's own iteration log")
    a = ap.parse_args()
    backend = ("cross-check" if a.cross_check
               else "mosek" if a.mosek
               else "none" if a.assemble_only
               else "cvxpy")
    run(N=a.patch, L=a.level, t=a.t, U=a.U, filling=a.nu,
        backend=backend, solver_log=a.verbose,
        skip_feasibility=a.skip_feasibility)
