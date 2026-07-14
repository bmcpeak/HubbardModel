"""Solver backends for SDPProblem.

Two backends consuming byte-identical problem data:

  solve_mosek  — production path, MOSEK Optimizer (Task) API, in the
                 hand-dualized standard form whose Schur complement scales
                 with the MOMENT COUNT rather than the matrix-entry count
                 (mandatory beyond level 2; the entry-wise formulation hits
                 ~240 GB at level 3). Requires a license; not executable in
                 the development container — run cross_check on first use.
  solve_cvxpy  — development/verification path, plain LMI form (cvxpy
                 dualizes internally). Fine at patch scales.

The optimum is a rigorous lower bound only up to solver tolerance; for
publishable numbers, extract the DUAL certificate and verify it
independently (planned: certificate checker in a later file).
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# cvxpy backend (verification)
# ----------------------------------------------------------------------

def solve_cvxpy(prob, solver: str = "CLARABEL", verbose: bool = False, **kw):
    """LMI form: y free, M_b(y) >> 0 directly. cvxpy dualizes internally."""
    import cvxpy as cp

    y = cp.Variable(prob.nvars)
    cons = []
    for row, rhs in prob.eq_constraints:
        cons.append(sum(c * y[k] for k, c in row.items()) == rhs)
    for b in prob.blocks:
        s = b.size
        rows = [[0] * s for _ in range(s)]
        for (i, j), row in b.entries.items():
            expr = sum(c * y[k] for k, c in row.items())
            rows[i][j] = expr
            if i != j:
                rows[j][i] = expr
        cons.append(cp.bmat(rows) >> 0)
    objective = cp.Minimize(sum(c * y[k] for k, c in prob.objective.items()))
    problem = cp.Problem(objective, cons)
    problem.solve(solver=solver, verbose=verbose, **kw)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"cvxpy status: {problem.status}")
    return problem.value, np.asarray(y.value), problem.status


# ----------------------------------------------------------------------
# MOSEK backend (production)  -- UNTESTED in this container (no license)
# ----------------------------------------------------------------------

def solve_mosek(prob, verbose: bool = False, tol: float = 1e-9):
    """Hand-dualized standard form — the scalable MOSEK formulation.

    Original (primal):  min c.y  s.t.  E y = d,  M_b(y) = sum_k y_k A^b_k >= 0.
    Fed to MOSEK (its primal = our dual):
        max d.lam  s.t.  sum_b <A^b_k, X_b> + (E^T lam)_k = c_k  (one row per
        moment k),  X_b >= 0,  lam free.
    Strong duality holds given a strictly feasible y — which the free-state
    feasibility check certifies (interior point with positive margin).

    Schur complement dimension = nvars (moment count), NOT the number of
    matrix entries; this is what makes level >= 3 solvable. The optimal y is
    recovered from the constraint duals; verify its sign convention against
    solve_cvxpy on a small problem on first use (cross_check does this).
    """
    import mosek
    import time

    inf = 0.0
    t0 = time.perf_counter()
    print(f"MOSEK: inverting {sum(len(b.entries) for b in prob.blocks)} "
          f"block entries over {prob.nvars} moments...", flush=True)

    # invert block entries: moment id k -> {block: [(i, j, coeff)]}
    inv = [dict() for _ in range(prob.nvars)]
    for bidx, b in enumerate(prob.blocks):
        for (i, j), row in b.entries.items():
            for k, c in row.items():
                inv[k].setdefault(bidx, []).append((i, j, c))

    nlam = len(prob.eq_constraints)
    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            if verbose:
                task.set_Stream(mosek.streamtype.log,
                                lambda s: print(s, end=""))
            task.appendvars(nlam)
            task.putvarboundsliceconst(0, nlam, mosek.boundkey.fr, -inf, +inf)
            task.appendbarvars([b.size for b in prob.blocks])
            task.appendcons(prob.nvars)

            for k in range(prob.nvars):
                if k and k % 1000 == 0:
                    print(f"MOSEK: constraint {k}/{prob.nvars} "
                          f"({time.perf_counter()-t0:.0f}s)", flush=True)
                for bidx, triples in inv[k].items():
                    dim = prob.blocks[bidx].size
                    rows = [max(i, j) for (i, j, c) in triples]
                    cols = [min(i, j) for (i, j, c) in triples]
                    # FULL coefficient values, no 0.5: the dual constraint
                    # needs <A_k, X> with A_k the actual symmetric
                    # coefficient matrix (A_ij = A_ji = row coefficient);
                    # MOSEK's lower-triangular inner product counts
                    # off-diagonals twice, which is exactly the symmetric
                    # inner product. (The 0.5 belonged to the old
                    # entry-PICKING formulation only; carrying it here
                    # halves all off-diagonal couplings — caught by
                    # cross_check as a too-low bound.)
                    vals = [c for (i, j, c) in triples]
                    mat = task.appendsparsesymmat(dim, rows, cols, vals)
                    task.putbaraij(k, bidx, [mat], [1.0])
                subs = [r for r, (row, _) in enumerate(prob.eq_constraints)
                        if k in row]
                if subs:
                    task.putarow(k, subs,
                                 [prob.eq_constraints[r][0][k] for r in subs])
                ck = prob.objective.get(k, 0.0)
                task.putconbound(k, mosek.boundkey.fx, ck, ck)

            for r, (_, rhs) in enumerate(prob.eq_constraints):
                task.putcj(r, rhs)
            task.putobjsense(mosek.objsense.maximize)
            print(f"MOSEK: emission done ({time.perf_counter()-t0:.0f}s), "
                  f"optimizing {prob.nvars} constraints x "
                  f"{len(prob.blocks)} PSD blocks "
                  f"(largest {max(b.size for b in prob.blocks)})...",
                  flush=True)

            task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, tol)
            task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, tol)
            task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, tol)

            task.optimize()
            solsta = task.getsolsta(mosek.soltype.itr)
            if solsta not in (mosek.solsta.optimal,):
                raise RuntimeError(f"MOSEK solution status: {solsta}")
            bound = task.getprimalobj(mosek.soltype.itr)
            y = np.zeros(prob.nvars)
            task.gety(mosek.soltype.itr, y)   # duals of the k-rows = moments
            # normalize the dual-sign convention using the first equality
            # (the identity moment, fixed to 1): formulation-agnostic
            row0, rhs0 = prob.eq_constraints[0]
            r = sum(c * y[k] for k, c in row0.items())
            if r * rhs0 < 0:
                y = -y
                r = -r
            if abs(r - rhs0) > 1e-4 * max(1.0, abs(rhs0)):
                print(f"  WARNING: recovered moment vector violates the "
                      f"identity constraint ({r} vs {rhs0}); treat y with "
                      f"suspicion (bound itself is unaffected)")
            return bound, y, str(solsta)


def cross_check(prob, tol=1e-5, verbose=False):
    """Solve with every available backend and compare. Run this on the
    first MOSEK use and after any solver-side change."""
    results = {}
    results["cvxpy/CLARABEL"] = solve_cvxpy(prob, verbose=verbose)
    try:
        results["mosek"] = solve_mosek(prob, verbose=verbose)
    except ImportError:
        print("  (mosek not installed; skipping)")
    vals = {name: r[0] for name, r in results.items()}
    names = list(vals)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            d = abs(vals[names[a]] - vals[names[b]])
            print(f"  {names[a]} = {vals[names[a]]:.8f}  vs  "
                  f"{names[b]} = {vals[names[b]]:.8f}   |diff| = {d:.2e}")
            assert d < tol, "backend disagreement"
    # feasibility diagnostics of each recovered moment vector
    for name, (bound, y, st) in results.items():
        diag = check_feasibility(prob, y, tol=1e-5)
        print(f"  {name}: recovered y -> min block eig "
              f"{diag['min_block_eig']:.2e}, eq viol {diag['eq_violation']:.2e}")
    return results


# ----------------------------------------------------------------------
# feasibility check of an explicit moment vector (no solver needed)
# ----------------------------------------------------------------------

def check_feasibility(prob, y, tol=1e-8):
    """Verify an explicit moment vector against every constraint.

    Returns dict of diagnostics; raises nothing. Used to certify that an
    exact physical state (e.g. free fermions) satisfies the assembled
    problem — the strongest available end-to-end sign check.
    """
    out = {"eq_violation": 0.0, "min_block_eig": np.inf, "worst_block": None}
    for row, rhs in prob.eq_constraints:
        v = abs(sum(c * y[k] for k, c in row.items()) - rhs)
        out["eq_violation"] = max(out["eq_violation"], v)
    for b in prob.blocks:
        M = np.zeros((b.size, b.size))
        for (i, j), row in b.entries.items():
            v = sum(c * y[k] for k, c in row.items())
            M[i, j] = M[j, i] = v
        w = np.linalg.eigvalsh(M)[0] if b.size else 0.0
        if w < out["min_block_eig"]:
            out["min_block_eig"] = w
            out["worst_block"] = f"{b.charges}/{b.label}"
    out["objective"] = sum(c * y[k] for k, c in prob.objective.items())
    out["feasible"] = (out["eq_violation"] < tol
                       and out["min_block_eig"] > -tol)
    return out
