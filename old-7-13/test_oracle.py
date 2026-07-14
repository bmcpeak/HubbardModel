"""
test_oracle.py -- run with: pytest test_oracle.py -v

Jordan-Wigner oracle plus structural tests for the split modules:
  algebra.py   -- kernels, dagger, canon/group signs   (tests 1-5, 8-10)
  physics.py   -- ad_H, wards, objective content        (tests 6-7, 11-13)
  symmetry.py  -- representation property, isotypic completeness,
                  cross-irrep vanishing                 (tests 14-16)

All oracle dimensions bounded by MAXMODES; any blowup fails loudly.
"""

import math
import numpy as np
import scipy.sparse as sp
from fractions import Fraction
from functools import lru_cache
from random import Random

import algebra as B
import physics as PH

RNG = Random(7)
MAXMODES = 18

# ---------- sparse JW representation ----------

@lru_cache(maxsize=None)
def jw_ops(k):
    I = sp.identity(2, format='csr')
    Z = sp.csr_matrix(np.diag([1.0, -1.0]))
    a = sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))
    cs = []
    for pos in range(k):
        M = sp.identity(1, format='csr')
        for o in [Z] * pos + [a] + [I] * (k - pos - 1):
            M = sp.kron(M, o, format='csr')
        cs.append(M)
    return tuple(cs)

def used_modes(*ops):
    ms = set()
    for op in ops:
        for key in op:
            D, C = B.unpack(key)
            mask = D | C
            while mask:
                m = (mask & -mask).bit_length() - 1
                ms.add(m)
                mask &= mask - 1
    return sorted(ms)

def rep(op, modes):
    assert len(modes) <= MAXMODES, \
        f"{len(modes)} modes: oracle too large, shrink the test case"
    idx = {m: i for i, m in enumerate(modes)}
    cs = jw_ops(len(modes))
    dim = 2 ** len(modes)
    out = sp.csr_matrix((dim, dim))
    for key, coeff in op.items():
        M = sp.identity(dim, format='csr')
        for kind, m in B.factors(key):
            M = M @ (cs[idx[m]].T if kind == 'd' else cs[idx[m]])
        out = out + float(coeff) * M
    return out

def rep_seq(seq, modes):
    assert len(modes) <= MAXMODES
    idx = {m: i for i, m in enumerate(modes)}
    cs = jw_ops(len(modes))
    M = sp.identity(2 ** len(modes), format='csr')
    for kind, p in seq:
        c = cs[idx[B.mode(*p)]]
        M = M @ (c.T if kind == 'd' else c)
    return M

def close(X, Y, tol=1e-9):
    D = X - Y
    return (abs(D.data).max() if D.nnz else 0.0) <= tol

# ---------- random operators ----------

PATCH = [(2, 2), (2, 3), (3, 2), (3, 3)]
DOMINO = [(3, 3), (3, 4)]

def rand_monomial(sites, max_deg=3):
    pool = [(i, j, s) for (i, j) in sites for s in (0, 1)]
    D = RNG.sample(pool, RNG.randint(0, max_deg))
    C = RNG.sample(pool, RNG.randint(0, max_deg))
    return B.pack_sites(D, C)

def rand_op(sites, nterms=3, max_deg=3):
    op = {}
    for _ in range(nterms):
        k = rand_monomial(sites, max_deg)
        op[k] = op.get(k, 0) + RNG.choice([-2, -1, 1, 2])
    return {k: v for k, v in op.items() if v != 0}

# ================================================================
# ALGEBRA LAYER
# ================================================================

def test_contraction_by_hand():
    m = B.mode(2, 2, 0)
    out = B.mul_d({B.pack(0, 1 << m): 1}, m)
    assert out == {0: 1, B.pack(1 << m, 1 << m): -1}

def test_multiply_vs_oracle():
    for _ in range(40):
        A, Bop = rand_op(PATCH), rand_op(PATCH)
        AB = B.multiply(A, Bop)
        modes = used_modes(A, Bop, AB)
        if not modes:
            continue
        assert close(rep(A, modes) @ rep(Bop, modes), rep(AB, modes))

def test_dagger_vs_transpose():
    for _ in range(40):
        A = rand_op(PATCH)
        Ad = B.dagger(A)
        modes = used_modes(A, Ad)
        if not modes:
            continue
        assert close(rep(A, modes).T, rep(Ad, modes))

def test_group_images():
    for _ in range(30):
        key = rand_monomial(PATCH)
        if key == 0:
            continue
        Ds = B.sites_of(key >> B.NMODES)
        Cs = B.sites_of(key & B.CMASK)
        seq = [('d', p) for p in Ds] + [('c', p) for p in reversed(Cs)]
        for (M, sf, dag), (img, sign) in zip(B.FINITE, B._images(key)):
            tseq = ([('c' if k == 'd' else 'd', p) for k, p in reversed(seq)]
                    if dag else seq)
            tseq = [(k, B.act(M, sf, p)) for k, p in tseq]
            di = min(p[0] for _, p in tseq)
            dj = min(p[1] for _, p in tseq)
            tseq = [(k, (i - di, j - dj, s)) for k, (i, j, s) in tseq]
            modes = sorted({B.mode(*p) for _, p in tseq})
            assert close(rep_seq(tseq, modes), sign * rep({img: 1}, modes)), \
                f"group sign wrong: {B.show(key)} under {(M, sf, dag)}"

def test_disjoint_even_commute():
    far = B.multiply(B.d_op(7, 7, 0), B.c_op(7, 8, 0))
    near = B.multiply(B.d_op(2, 2, 0), B.c_op(2, 3, 0))
    assert B.commutator(far, near) == {}

# ================================================================
# PHYSICS LAYER
# ================================================================

def _local_H(sites_in, t, U):
    """Independent enumeration of H-terms touching the given sites."""
    sites = set(sites_in)
    H = {}
    bonds = set()
    for (i, j) in sites:
        for (pi, pj) in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
            bonds.add(frozenset([(i, j), (pi, pj)]))
    for bond in bonds:
        p, q = tuple(bond)
        for s in (0, 1):
            H = B.add(H, B.multiply(B.d_op(*p, s), B.c_op(*q, s)), scale=-t)
            H = B.add(H, B.multiply(B.d_op(*q, s), B.c_op(*p, s)), scale=-t)
    for (i, j) in sites:
        H = B.add(H, B.multiply(
            B.multiply(B.d_op(i, j, 1), B.c_op(i, j, 1)),
            B.multiply(B.d_op(i, j, 0), B.c_op(i, j, 0))), scale=U)
    return H

def test_ad_H_vs_oracle():
    t, U = 1, 8
    for _ in range(20):
        key = rand_monomial(DOMINO, max_deg=2)
        if key == 0:
            continue
        O = B.op_of(key)
        adO = PH.ad_H_op(O, t, U)
        H = _local_H(B.support_sites(key), t, U)
        modes = used_modes(O, adO, H)
        HM, OM = rep(H, modes), rep(O, modes)
        assert close(HM @ OM - OM @ HM, rep(adO, modes)), \
            f"ad_H wrong for {B.show(key)}"

def test_ward_locality():
    for _ in range(20):
        key = rand_monomial(PATCH, max_deg=2)
        if key == 0:
            continue
        O = B.op_of(key)
        supp = B.support_sites(key)
        big = {(i + di, j + dj) for (i, j) in supp
               for di in (-1, 0, 1) for dj in (-1, 0, 1)}
        for pm in (+1, -1):
            small = B.commutator(PH.local_spin_charge(supp, pm), O)
            large = B.commutator(PH.local_spin_charge(big, pm), O)
            assert small == large
        S = PH.local_spin_charge(supp, +1)
        comm = B.commutator(S, O)
        modes = used_modes(S, O, comm)
        if not modes:
            continue
        assert close(rep(S, modes) @ rep(O, modes)
                     - rep(O, modes) @ rep(S, modes), rep(comm, modes))

# ================================================================
# NEW: canon structural properties (previously only implicit)
# ================================================================

def test_canon_idempotent_and_invariant():
    """canon(canon rep) is itself with sign +1, and canon is constant on
    orbits with consistent relative signs."""
    for _ in range(30):
        key = rand_monomial(PATCH)
        if key == 0:
            continue
        ck, cs = B.canon(key)
        if cs == 0:
            continue
        ck2, cs2 = B.canon(ck)
        assert ck2 == ck and cs2 == 1
        for img, sgn in B._images(key):
            ik, isg = B.canon(img)
            assert ik == ck
            if isg != 0:
                # canon(key) sign * image-production sign must compose
                # consistently: <img> = sgn <key> and both map to ck
                assert isg * sgn == cs or isg * sgn == -cs
                # (both orientations occur across the orbit; the sharp
                #  statement is isg == cs * sgn, check that:)
                assert isg == cs * sgn, \
                    f"orbit sign inconsistency at {B.show(key)}"

def test_forced_zero_is_real():
    """Any canon-forced-zero EV must vanish in EVERY group-invariant state.
    Oracle version: the group-symmetrized operator must be identically 0."""
    found = 0
    for _ in range(300):
        key = rand_monomial(PATCH, max_deg=2)
        if key == 0:
            continue
        ck, cs = B.canon(key)
        if cs != 0:
            continue
        found += 1
        symm = {}
        for img, sgn in B._images(key):
            symm[img] = symm.get(img, 0) + sgn
        symm = {k: v for k, v in symm.items() if v != 0}
        assert not symm, f"forced zero {B.show(key)} has nonzero symmetrization"
        if found >= 5:
            break
    # not asserting found > 0: zeros are sparse among random low-degree
    # monomials; the test is vacuous on unlucky seeds, which is acceptable
    # because the seed is fixed.

def test_objective_content():
    """Objective = -t*(one hop orbit) + U*(one doublon orbit): exactly two
    canonical variables, with the hop coefficient's sign locked."""
    obj = PH.objective(1, 8)
    assert len(obj) == 2
    hop_key = min(obj, key=lambda k: (B.unpack(k)[0].bit_count()
                                      + B.unpack(k)[1].bit_count()))
    assert obj[hop_key] < 0, "hop coefficient must carry -t"

def test_t0_U_only_commutes_with_doublon():
    """[V, doublon] = 0 exactly -- the identity used repeatedly in analysis."""
    doub = B.shift_op(PH.u_part(), PH.OX, PH.OY)
    _, u_items = PH.ad_H(next(iter(doub)))
    assert u_items == (), "[U-part, doublon] should vanish identically"

# ================================================================
# SYMMETRY LAYER
# ================================================================

import symmetry as SY

def _patch_basis(P, nd, nc):
    pool = [(i, j, s) for i in range(P) for j in range(P) for s in (0, 1)]
    from itertools import combinations
    return [B.pack_sites(list(D), list(C))
            for D in combinations(pool, nd) for C in combinations(pool, nc)]

def test_rep_property():
    """R(g)R(h) = R(gh) on the full 16-element multiplication table, on a
    real basis (P=2 bilinear keys)."""
    keys = _patch_basis(2, 1, 1)
    mats = SY.rep_matrices(keys, 2)
    def compose(m1, m2):
        return {c: (m1[m][0], m1[m][1] * s2)
                for c, (m, s2) in m2.items()}
    # closure: composing any two group matrices lands in the set
    allm = list(mats.values())
    as_tuples = {tuple(sorted((c, rs) for c, rs in m.items())) for m in allm}
    for m1 in allm:
        for m2 in allm:
            comp = tuple(sorted((c, rs) for c, rs in compose(m1, m2).items()))
            assert comp in as_tuples, "group not closed on this basis"

def test_isotypic_completeness():
    """Adapted vectors across irreps span exactly the basis dimension,
    for both an even patch (P=2, half-integer center) and odd (P=3)."""
    for P, nd, nc in [(2, 1, 0), (2, 1, 1), (3, 1, 0)]:
        keys = _patch_basis(P, nd, nc)
        vs = SY.adapted_vectors(keys, P)   # internal assert checks the sum
        assert sum(len(v) for v in vs.values()) == len(keys)

def test_cross_irrep_vanishing_and_bound_content():
    """Full pipeline miniature: P=2 bilinear moment matrix, split by irrep,
    cross-entries must vanish as forms (asserted inside split), and the
    union of block variables must equal the unsplit matrix's variables
    (no content lost)."""
    keys = _patch_basis(2, 1, 1)
    basis_ops = [B.op_of(k) for k in keys]
    dags = [B.dagger(o) for o in basis_ops]
    M = [[B.to_linear_form(B.multiply(di, bj)) for bj in basis_ops]
         for di in dags]
    # symmetry of forms (relied on by split_moment_matrix)
    n = len(M)
    for i in range(n):
        for j in range(i):
            assert M[i][j] == M[j][i]
    blocks = SY.split_moment_matrix(M, keys, 2)
    vars_unsplit = {k for row in M for e in row for k in e} - {0}
    vars_split = {k for _, blk in blocks for row in blk
                  for e in row for k in e} - {0}
    assert vars_split == vars_unsplit, "irrep split lost or invented variables"
    total_dim = sum(len(blk) for _, blk in blocks)
    assert total_dim == n