# test_oracle.py -- run with: pytest test_oracle.py -v
#
# Jordan-Wigner oracle for the bitmask fermion algebra. Every test checks
# bootstrap_hubbard against a faithful sparse-matrix representation.
# All dimensions are bounded by construction; MAXMODES makes any future
# blowup a loud failure instead of a hang.

import numpy as np
import scipy.sparse as sp
from functools import lru_cache
from random import Random
from old import bootstrap_hubbard as B

RNG = Random(7)
MAXMODES = 18

# ---------- sparse JW representation ----------

@lru_cache(maxsize=None)
def jw_ops(k):
    """Annihilators c_0..c_{k-1} on a k-mode Fock space, CSR."""
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
    """Sparse matrix of an operator dict on the given mode set."""
    assert len(modes) <= MAXMODES, \
        f"{len(modes)} modes: oracle dimension too large, shrink the test case"
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
    """Matrix of an ordered elementary-factor sequence [(kind,(i,j,s)),...]."""
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

PATCH = [(2, 2), (2, 3), (3, 2), (3, 3)]      # interior 2x2
DOMINO = [(3, 3), (3, 4)]                     # interior bond

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

# ---------- 1. hand-checkable contraction ----------

def test_contraction_by_hand():
    # c0 * d0 = 1 - d0 c0, on mode of site (2,2) spin 0
    m = B.mode(2, 2, 0)
    out = B.mul_d({B.pack(0, 1 << m): 1}, m)
    assert out == {0: 1, B.pack(1 << m, 1 << m): -1}

# ---------- 2. products ----------

def test_multiply_vs_oracle():
    for _ in range(40):
        A, Bop = rand_op(PATCH), rand_op(PATCH)
        AB = B.multiply(A, Bop)
        modes = used_modes(A, Bop, AB)
        if not modes:
            continue
        assert close(rep(A, modes) @ rep(Bop, modes), rep(AB, modes))

# ---------- 3. dagger is transpose, sign-free ----------

def test_dagger_vs_transpose():
    for _ in range(40):
        A = rand_op(PATCH)
        Ad = B.dagger(A)
        modes = used_modes(A, Ad)
        if not modes:
            continue
        assert close(rep(A, modes).T, rep(Ad, modes))

# ---------- 4. group action signs, all 32 elements ----------

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

# ---------- 5. disjoint even terms commute (the locality principle) ----------

def test_disjoint_even_commute():
    far = B.multiply(B.d_op(7, 7, 0), B.c_op(7, 8, 0))
    near = B.multiply(B.d_op(2, 2, 0), B.c_op(2, 3, 0))
    comm = B.commutator(far, near)
    assert comm == {}
    modes = used_modes(far, near)
    F, N = rep(far, modes), rep(near, modes)
    assert close(F @ N - N @ F, sp.csr_matrix((2 ** len(modes),) * 2))

# ---------- 6. ad_H term content, domino supports ----------

def _local_H(omodes_sites, t, U):
    """H-terms sharing a site with the given set (independent enumeration:
    by site adjacency, not ad_H's bond machinery)."""
    sites = set(omodes_sites)
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
        adO = B.ad_H_op(O, t, U)
        H = _local_H(B.support_sites(key), t, U)
        modes = used_modes(O, adO, H)
        HM, OM = rep(H, modes), rep(O, modes)
        assert close(HM @ OM - OM @ HM, rep(adO, modes)), \
            f"ad_H wrong for {B.show(key)}"

def test_ad_H_doublon():
    """The sanity_check_G object, certified: [H, doublon] against matrices."""
    t, U = 1, 8
    doub = B.shift_op(B.u_part(), B.OX, B.OY)
    adO = B.ad_H_op(doub, t, U)
    H = _local_H(B.support_sites(next(iter(doub))), t, U)
    modes = used_modes(doub, adO, H)
    HM, OM = rep(H, modes), rep(O := rep(doub, modes), modes) if False else (rep(H, modes), rep(doub, modes))
    HM, OM = rep(H, modes), rep(doub, modes)
    assert close(HM @ OM - OM @ HM, rep(adO, modes))
    # and the G diagonal itself, pre-canon: <doub^dag [H,doub]> as an operator
    entry = B.multiply(B.dagger(doub), adO)
    assert close(rep(B.dagger(doub), modes) @ rep(adO, modes),
                 rep(entry, modes))

# ---------- 7. Ward locality: exact cancellation + one certification ----------

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
            small = B.commutator(B.local_spin_charge(supp, pm), O)
            large = B.commutator(B.local_spin_charge(big, pm), O)
            assert small == large, \
                "disjoint S-terms failed to cancel exactly: sign bug in multiply"
        S = B.local_spin_charge(supp, +1)
        comm = B.commutator(S, O)
        modes = used_modes(S, O, comm)
        if not modes:
            continue
        assert close(rep(S, modes) @ rep(O, modes) - rep(O, modes) @ rep(S, modes),
                     rep(comm, modes))