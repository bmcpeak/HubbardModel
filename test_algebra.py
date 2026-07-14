"""Verify algebra.py against a dense Jordan-Wigner representation.

Strategy: build explicit 2^n x 2^n matrices for a_i satisfying the CAR,
map every canonical monomial to the corresponding matrix product, and check
that Operator arithmetic is a *-algebra homomorphism. Any sign error in
mul_c / mul_d fails these tests loudly.

Run:  python -m pytest tests/test_algebra.py -q      (or just python tests/test_algebra.py)
"""

import random
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from algebra import (Operator, mono_mul, mono_dag, mono_str, bits_ascending,
                     bits_descending, commutator, dop, cop, nop, IDENTITY_MONO)

N_MODES = 6          # 64-dim dense space; big enough for nontrivial overlaps
N_RANDOM = 400
random.seed(20260713)


# ----------------------------------------------------------------------
# dense Jordan-Wigner reference (independent sign bookkeeping)
# ----------------------------------------------------------------------

def jw_annihilators(n):
    I = np.eye(2)
    Z = np.diag([1.0, -1.0])
    A = np.array([[0.0, 1.0], [0.0, 0.0]])   # |occ> -> |empty>, basis (empty, occ)
    ops = []
    for i in range(n):
        mats = [Z] * i + [A] + [I] * (n - i - 1)
        M = mats[0]
        for m in mats[1:]:
            M = np.kron(M, m)
        ops.append(M)
    return ops


A_MATS = jw_annihilators(N_MODES)
D_MATS = [a.conj().T for a in A_MATS]
DIM = 2 ** N_MODES
ID = np.eye(DIM)


def mono_matrix(m):
    """Matrix of the canonical string: daggers ascending, then c's descending."""
    M = ID
    for i in bits_ascending(m[0]):
        M = M @ D_MATS[i]
    for i in bits_descending(m[1]):
        M = M @ A_MATS[i]
    return M


def op_matrix(op: Operator):
    M = np.zeros((DIM, DIM))
    for m, v in op.terms.items():
        M = M + float(v) * mono_matrix(m)
    return M


def rand_mono():
    dag = random.getrandbits(N_MODES)
    c = random.getrandbits(N_MODES)
    return (dag, c)


def rand_operator(nterms=3, coeff_pool=(-2, -1, 1, 2, 3)):
    t = {}
    for _ in range(nterms):
        t[rand_mono()] = random.choice(coeff_pool)
    return Operator(t)


def assert_close(M1, M2, msg=""):
    assert np.max(np.abs(M1 - M2)) < 1e-9, msg


# ----------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------

def test_reference_car():
    """Sanity of the reference itself: {a_i, a†_j} = δ_ij, {a_i, a_j} = 0."""
    for i in range(N_MODES):
        for j in range(N_MODES):
            anti = A_MATS[i] @ D_MATS[j] + D_MATS[j] @ A_MATS[i]
            assert_close(anti, (ID if i == j else 0 * ID), f"CAR fail {i},{j}")
            anti2 = A_MATS[i] @ A_MATS[j] + A_MATS[j] @ A_MATS[i]
            assert_close(anti2, 0 * ID, f"aa CAR fail {i},{j}")


def test_car_at_operator_level():
    """Our algebra reproduces the CAR exactly (integer coefficients)."""
    for i in range(N_MODES):
        for j in range(N_MODES):
            anti = cop(i) * dop(j) + dop(j) * cop(i)
            expect = Operator.identity() if i == j else Operator.zero()
            assert anti == expect, f"CAR fail at ({i},{j}): {anti!r}"
            assert (cop(i) * cop(j) + cop(j) * cop(i)) == Operator.zero()
            assert (dop(i) * dop(j) + dop(j) * dop(i)) == Operator.zero()


def test_canonical_generator_ordering():
    """Building strings out of generators lands in canonical form with right signs."""
    # d1 d0 = -d0 d1 ; canonical key stores ascending daggers
    assert (dop(1) * dop(0)) == Operator({(0b11, 0): -1})
    assert (dop(0) * dop(1)) == Operator({(0b11, 0): 1})
    # c0 c1 = -c1 c0 ; canonical stores descending c's, i.e. key (0, 0b11) means c1 c0
    assert (cop(1) * cop(0)) == Operator({(0, 0b11): 1})
    assert (cop(0) * cop(1)) == Operator({(0, 0b11): -1})
    # number operator
    assert (dop(2) * cop(2)) == nop(2)
    assert (cop(2) * dop(2)) == Operator.identity() - nop(2)


def test_random_monomial_products():
    """mono_mul agrees with dense matrices on random monomial pairs."""
    for _ in range(N_RANDOM):
        m1, m2 = rand_mono(), rand_mono()
        prod = mono_mul(m1, m2)
        lhs = mono_matrix(m1) @ mono_matrix(m2)
        rhs = np.zeros((DIM, DIM))
        for key, s in prod.items():
            rhs = rhs + s * mono_matrix(key)
        assert_close(lhs, rhs, f"product mismatch: {mono_str(m1)} * {mono_str(m2)}")


def test_operator_products_and_adjoint():
    for _ in range(60):
        A, B = rand_operator(), rand_operator()
        assert_close(op_matrix(A * B), op_matrix(A) @ op_matrix(B), "op product")
        assert_close(op_matrix(A.dag()), op_matrix(A).conj().T, "adjoint")
        assert_close(op_matrix(commutator(A, B)),
                     op_matrix(A) @ op_matrix(B) - op_matrix(B) @ op_matrix(A),
                     "commutator")


def test_adjoint_is_signless():
    """(dag, c)† = (c, dag) with sign exactly +1, for random monomials."""
    for _ in range(N_RANDOM):
        m = rand_mono()
        assert_close(mono_matrix(m).conj().T, mono_matrix(mono_dag(m)),
                     f"adjoint sign fail on {mono_str(m)}")


def test_associativity_exact():
    """(A*B)*C == A*(B*C) with exact integer arithmetic, no matrices."""
    for _ in range(40):
        A, B, C = rand_operator(), rand_operator(), rand_operator()
        assert (A * B) * C == A * (B * C)


def test_two_site_hubbard():
    """Mini Hubbard on 2 sites (4 modes): H hermitian, [H,H]=0, matrix match.

    mode = 2*site + spin; sites 0,1; spins 0 (up), 1 (down).
    H = -t Σ_σ (a†_{0σ} a_{1σ} + h.c.) + U Σ_x n_{x↑} n_{x↓}
    """
    t, U = 1, 8

    def mode(x, s):
        return 2 * x + s

    H = Operator.zero()
    for s in (0, 1):
        hop = dop(mode(0, s)) * cop(mode(1, s))
        H = H + (-t) * (hop + hop.dag())
    for x in (0, 1):
        H = H + U * (nop(mode(0 if x == 0 else 1, 0)) * nop(mode(x, 1)))

    assert H.is_hermitian()
    assert commutator(H, H) == Operator.zero()

    # dense check on the first 4 modes
    Hm = op_matrix(H)
    assert_close(Hm, Hm.conj().T, "dense H not hermitian")

    # a physics number: half-filled ground state energy of 2-site Hubbard,
    # E0 = (U - sqrt(U^2 + 16 t^2)) / 2, valid in the (N_up, N_dn) = (1,1)
    # sector — the full Fock space has lower sectors (N=1 bonding at -t),
    # so project first.
    n_up = op_matrix(nop(mode(0, 0)) + nop(mode(1, 0)))
    n_dn = op_matrix(nop(mode(0, 1)) + nop(mode(1, 1)))
    diag_up = np.round(np.diag(n_up)).astype(int)
    diag_dn = np.round(np.diag(n_dn)).astype(int)
    sel = np.where((diag_up == 1) & (diag_dn == 1))[0]
    w = np.linalg.eigvalsh(Hm[np.ix_(sel, sel)])
    e_exact = (U - np.sqrt(U * U + 16 * t * t)) / 2
    assert abs(w[0] - e_exact) < 1e-9, f"2-site GS energy {w[0]} vs {e_exact}"


def test_pauli_zeros():
    assert (dop(3) * dop(3)) == Operator.zero()
    assert (cop(3) * cop(3)) == Operator.zero()
    m = (0b101, 0b011)
    assert mono_mul(m, (0b100, 0)) == {}  # dagger already present, pass-through killed
    # ... but contraction can survive even when pass-through dies:
    prod = mono_mul((0b1, 0b1), (0b1, 0))   # (d0 c0)(d0) = d0 (c0 d0) = d0 (1 - d0 c0) = d0
    assert prod == {(0b1, 0): 1}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"ok  {f.__name__}")
    print(f"\nall {len(fns)} tests passed")
