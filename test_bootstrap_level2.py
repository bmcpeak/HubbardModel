"""End-to-end tests of the minimal bootstrap slice (3x3, level 2).

Ordered by what they pin down:
  1. Wick determinant convention, spinless: against applier-based ED of a
     free OBC system (independent of the Wick code path);
  2. spinful factorization + crossing sign: same, mixed-spin monomials;
  3. FREE-STATE FEASIBILITY: exact closed-shell Slater moments on a big
     torus satisfy EVERY assembled constraint. This exercises algebra,
     symmetry, moments, and assembly signs simultaneously against an
     exact many-body state; it is the test that would catch a sign error
     nothing else catches.
  4. solved bounds: U=0 bound vs exact free energy (must be <=, should be
     close); U=8 bound vs free-state variational energy (must be <=).

Run: python tests/test_bootstrap_level2.py    (takes a few minutes)
"""

import random
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from algebra import Operator
from modes import Lattice
from hamiltonian import Hubbard
from symmetry import PointGroup, decompose
from moments import MomentTable
from basis_patch import patch_basis, patch_center
from assemble import assemble_problem
from free_fermions import TorusGaussianState, WickEvaluator
from sdp import check_feasibility, solve_cvxpy
from test_symmetry import apply_mono_state

random.seed(20260713)


# ----------------------------------------------------------------------
# applier-based free ED (independent of the Wick code path)
# ----------------------------------------------------------------------

def slater_state(orbitals, mode_of_site):
    """State dict {occ_mask: amp} for filled orbitals, via our applier.

    orbitals: list of coefficient vectors over sites; mode_of_site maps a
    site index to a mode bit position.
    """
    state = {0: 1.0}
    for phi in orbitals:
        new = {}
        for f, a in state.items():
            for site, coef in enumerate(phi):
                if abs(coef) < 1e-15:
                    continue
                r = apply_mono_state((1 << mode_of_site(site), 0), f)
                if r is not None:
                    key = r[1]
                    new[key] = new.get(key, 0.0) + coef * r[0] * a
        state = new
    # normalize (orbitals orthonormal => should already be 1)
    nrm = np.sqrt(sum(a * a for a in state.values()))
    return {f: a / nrm for f, a in state.items()}


def state_expval(state, op: Operator):
    out = 0.0
    for f, a in state.items():
        for m, v in op.terms.items():
            r = apply_mono_state(m, f)
            if r is not None and r[1] in state:
                out += state[r[1]] * float(v) * r[0] * a
    return out


def obc_free_orbitals(n_sites_1d, n_fill, hop_pairs):
    """Lowest n_fill orbitals of a free OBC hopping Hamiltonian."""
    ns = n_sites_1d
    H1 = np.zeros((ns, ns))
    for (a, b) in hop_pairs:
        H1[a, b] = H1[b, a] = -1.0
    w, V = np.linalg.eigh(H1)
    assert w[n_fill] - w[n_fill - 1] > 1e-10, "degenerate Fermi level in test setup"
    return [V[:, k] for k in range(n_fill)], V, w


def test_wick_spinless():
    """1D open chain, one spin species: Wick det == applier ED."""
    ns, nf = 7, 3
    pairs = [(i, i + 1) for i in range(ns - 1)]
    orbs, V, w = obc_free_orbitals(ns, nf, pairs)
    G = V[:, :nf] @ V[:, :nf].T           # <d_i c_j>
    state = slater_state(orbs, lambda s: 2 * s)   # spin-up modes of a 1D lat

    lat = Lattice(7)                       # sites (x, 0), spin 0
    ev = WickEvaluator(lat, lambda a, b: G[a[0], b[0]])
    for _ in range(200):
        p = random.randint(1, 3)
        ds = sorted(random.sample(range(ns), p))
        cs = sorted(random.sample(range(ns), p))
        m = (sum(1 << lat.mode(x, 0, 0) for x in ds),
             sum(1 << lat.mode(x, 0, 0) for x in cs))
        direct = state_expval(state, Operator({m: 1}))
        wick = ev.mono_expval(m)
        assert abs(direct - wick) < 1e-9, \
            f"spinless Wick mismatch: {direct} vs {wick}"


def test_wick_spinful():
    """Two species on a 1D chain: factorization + crossing sign."""
    ns, nf = 5, 2
    pairs = [(i, i + 1) for i in range(ns - 1)]
    orbs, V, w = obc_free_orbitals(ns, nf, pairs)
    G = V[:, :nf] @ V[:, :nf].T
    lat = Lattice(5)

    # spinful Slater state: fill the same orbitals in both species
    state = {0: 1.0}
    for spin in (0, 1):
        for phi in orbs:
            new = {}
            for f, a in state.items():
                for x, coef in enumerate(phi):
                    if abs(coef) < 1e-15:
                        continue
                    r = apply_mono_state((1 << lat.mode(x, 0, spin), 0), f)
                    if r is not None:
                        new[r[1]] = new.get(r[1], 0.0) + coef * r[0] * a
            state = new
    nrm = np.sqrt(sum(a * a for a in state.values()))
    state = {f: a / nrm for f, a in state.items()}

    ev = WickEvaluator(lat, lambda a, b: G[a[0], b[0]])
    for _ in range(200):
        dag = 0
        c = 0
        for spin in (0, 1):
            p = random.randint(0, 2)
            for x in random.sample(range(ns), p):
                dag |= 1 << lat.mode(x, 0, spin)
            for x in random.sample(range(ns), p):
                c |= 1 << lat.mode(x, 0, spin)
        m = (dag, c)
        if m == (0, 0):
            continue
        direct = state_expval(state, Operator({m: 1}))
        wick = ev.mono_expval(m)
        assert abs(direct - wick) < 1e-9, \
            f"spinful Wick mismatch: {direct} vs {wick} on {m}"


# ----------------------------------------------------------------------
# the big one: exact-state feasibility of the assembled problem
# ----------------------------------------------------------------------

def build_level2_problem(U=8, filling=7 / 8):
    N, Lmax = 3, 2
    lat = Lattice(N)
    hub = Hubbard(lat, t=1, U=U)
    ops = patch_basis(lat, 0, 0, N, Lmax)
    pg = PointGroup(lat, patch_center(0, 0, N), include_flip=True)
    blocks = decompose(pg, ops)
    table = MomentTable(lat)
    prob = assemble_problem(blocks, table, hub, (1, 1), filling)
    return prob, table, lat


def test_free_state_feasibility():
    tor = TorusGaussianState(64, (7 / 8) / 2)   # EXACT filling now
    nu = 2 * tor.nu_sigma
    assert abs(nu - 7 / 8) < 1e-12
    prob, table, lat = build_level2_problem(U=8, filling=nu)
    wick = WickEvaluator(lat, lambda a, b: tor.g(a[0] - b[0], a[1] - b[1]))
    y = wick.moment_vector(table)
    diag = check_feasibility(prob, y, tol=1e-7)
    print(f"    free-state check: eq viol {diag['eq_violation']:.2e}, "
          f"min block eig {diag['min_block_eig']:.2e} "
          f"(worst {diag['worst_block']}), nu={nu:.5f}")
    assert diag["feasible"], f"exact free state INFEASIBLE: {diag}"
    # and its objective value is the known variational energy
    e_var = 2 * tor.e_kin_per_site_per_spin + 8 * wick.double_occupancy()
    assert abs(diag["objective"] - e_var) < 1e-8


def test_fast_path_equals_generic():
    """One-sided seeded assembly == generic dag-product assembly, entrywise."""
    from assemble import assemble_block
    from moments import MomentTable
    lat = Lattice(3)
    ops = patch_basis(lat, 0, 0, 3, 2)
    pg = PointGroup(lat, patch_center(0, 0, 3), include_flip=True)
    blocks = decompose(pg, ops)
    t1, t2 = MomentTable(lat), MomentTable(lat)
    for b in blocks:
        fast = assemble_block(b, t1)
        b_generic = type(b)(b.charges, b.irrep, ops=b.ops, seeds=None)
        gen = assemble_block(b_generic, t2)
        assert set(fast.entries) == set(gen.entries), f"entry sets differ in {b}"
        for key in fast.entries:
            fr, gr = fast.entries[key], gen.entries[key]
            assert set(fr) == set(gr)
            for k in fr:
                assert abs(fr[k] - gr[k]) < 1e-12, f"entry mismatch {b} {key}"


def test_solved_bounds():
    tor = TorusGaussianState(64, (7 / 8) / 2)
    nu = 2 * tor.nu_sigma
    e_free = 2 * tor.e_kin_per_site_per_spin

    # U=0: bound must sit below the exact free energy density
    prob0, _, _ = build_level2_problem(U=0, filling=nu)
    b0, y0, st0 = solve_cvxpy(prob0)
    print(f"    U=0 bound {b0:.6f} vs exact free {e_free:.6f} "
          f"(gap {e_free - b0:.6f}) [{st0}]")
    assert b0 <= e_free + 1e-6

    # U=8: bound must sit below the free-state variational energy
    wick = WickEvaluator(Lattice(3), lambda a, b: tor.g(a[0] - b[0], a[1] - b[1]))
    e_var = e_free + 8 * wick.double_occupancy()
    prob8, _, _ = build_level2_problem(U=8, filling=nu)
    b8, y8, st8 = solve_cvxpy(prob8)
    print(f"    U=8 bound {b8:.6f} vs free-state variational {e_var:.6f} [{st8}]")
    assert b8 <= e_var + 1e-6
    # interaction must cost something: U=8 bound above U=0 bound
    assert b8 >= b0 - 1e-6



def test_dual_formulation_matches_primal():
    """The hand-dualized form (exactly the data solve_mosek constructs,
    including the full-coefficient symmetric inner product) must reproduce
    the primal LMI optimum. Guards the MOSEK backend's math without
    needing a MOSEK license; caught the halved-off-diagonal bug of
    2026-07 (bound -2.43 instead of -1.12)."""
    import cvxpy as cp
    prob, table, lat = build_level2_problem(U=8, filling=7 / 8)
    b_primal, _, _ = solve_cvxpy(prob)

    inv = [dict() for _ in range(prob.nvars)]
    for bidx, b in enumerate(prob.blocks):
        for (i, j), row in b.entries.items():
            for k, c in row.items():
                inv[k].setdefault(bidx, []).append((i, j, c))
    Xs = [cp.Variable((b.size, b.size), symmetric=True) for b in prob.blocks]
    lam = cp.Variable(len(prob.eq_constraints))
    cons = [X >> 0 for X in Xs]
    for k in range(prob.nvars):
        expr = 0
        for bidx, triples in inv[k].items():
            for (i, j, c) in triples:
                expr = expr + (c * Xs[bidx][i, i] if i == j
                               else 2 * c * Xs[bidx][i, j])
        for r, (row, _) in enumerate(prob.eq_constraints):
            if k in row:
                expr = expr + row[k] * lam[r]
        cons.append(expr == prob.objective.get(k, 0.0))
    d = np.array([rhs for _, rhs in prob.eq_constraints])
    dual = cp.Problem(cp.Maximize(d @ lam), cons)
    dual.solve(solver="CLARABEL")
    assert abs(dual.value - b_primal) < 1e-5, \
        f"dual {dual.value} != primal {b_primal}"
    print(f"    dual form = {dual.value:.8f}, primal = {b_primal:.8f}")


if __name__ == "__main__":
    for f in (test_wick_spinless, test_wick_spinful,
              test_free_state_feasibility, test_fast_path_equals_generic,
              test_dual_formulation_matches_primal, test_solved_bounds):
        f()
        print(f"ok  {f.__name__}")
    print("\nall bootstrap tests passed")
