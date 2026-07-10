"""
symmetry.py -- point-group irrep decomposition of operator bases.

Group: G = D4 x Z2(spinflip), |G| = 16, acting on monomial bases as signed
permutations (dagger excluded: it maps charge q to -q, lives in canon).

Paths:
  split_moment_matrix   -- exact Fraction reference (slow; pytest uses it)
  split_to_blockdata    -- production: triplet extraction (single numpy-
                           sorted pass), integer conjugation batched per
                           irrep, A1-attached identity, E-multiplicity dedup.
                           Stage timings printed via log.

Validity: cross-irrep entries vanish by Schur GIVEN M commutes with the
representation; _check_equivariance_fast verifies that premise exactly
(sorted-triplet comparison, no tolerances). E-dedup: the defining 2D rep of
D4 IS E, so the row projector P_00 = (2/16) sum_g D(g)_00 chi_z2 R(g)
extracts one partner copy; the dropped partner's block equals it by Schur.
"""

import time
from fractions import Fraction
from math import gcd
import numpy as np
import scipy.sparse as sp

from algebra import (PG8, mode, sites_of, pack_sites, unpack, parity_sign)

# ============================================================
# Character table of D4 x Z2
# ============================================================

_D4_CLASSES = [(0,), (2,), (1, 3), (4, 5), (6, 7)]
_D4_CHARS = {
    "A1": (1, 1, 1, 1, 1),
    "A2": (1, 1, 1, -1, -1),
    "B1": (1, 1, -1, 1, -1),
    "B2": (1, 1, -1, -1, 1),
    "E":  (2, -2, 0, 0, 0),
}
_DIMS = {"A1": 1, "A2": 1, "B1": 1, "B2": 1, "E": 2}

def _char(d4, pg):
    for cls, chi in zip(_D4_CLASSES, _D4_CHARS[d4]):
        if pg in cls:
            return chi
    raise ValueError(pg)

IRREPS = [(d4, z2) for d4 in _D4_CHARS for z2 in (0, 1)]

# ============================================================
# Group action on a monomial basis
# ============================================================

def _act_centered(pg_index, sf, key, P):
    """Patch-centered action via doubled coordinates (center at 0 for any
    parity of P). Returns (image_key, sign)."""
    (a, b), (c_, d_) = PG8[pg_index]
    def f(p):
        i, j, s = p
        u, v = 2 * i - (P - 1), 2 * j - (P - 1)
        u2, v2 = a * u + b * v, c_ * u + d_ * v
        assert (u2 + (P - 1)) % 2 == 0
        i2, j2 = (u2 + (P - 1)) // 2, (v2 + (P - 1)) // 2
        assert 0 <= i2 < P and 0 <= j2 < P, "patch-centered action escaped"
        return (i2, j2, s ^ sf)
    D, C = unpack(key)
    Ds = [f(p) for p in sites_of(D)]
    Cs = [f(p) for p in sites_of(C)]
    sign = parity_sign(Ds) * parity_sign(Cs)
    return pack_sites(Ds, Cs), sign

def rep_matrices(basis_keys, P):
    """{(pg, sf): {col: (row, sign)}} for all 16 elements; asserts closure
    of the basis under the group."""
    index = {k: i for i, k in enumerate(basis_keys)}
    assert len(index) == len(basis_keys), "duplicate basis elements"
    mats = {}
    for pg in range(8):
        for sf in (0, 1):
            col_map = {}
            for col, k in enumerate(basis_keys):
                img, sgn = _act_centered(pg, sf, k, P)
                assert img in index, "basis not closed under the group"
                col_map[col] = (index[img], sgn)
            mats[(pg, sf)] = col_map
    return mats

def _check_representation(mats):
    def compose(m1, m2):
        return {c: (m1[m][0], m1[m][1] * s2) for c, (m, s2) in m2.items()}
    assert compose(mats[(1, 0)], mats[(1, 0)]) == mats[(2, 0)], \
        "R(90)^2 != R(180)"

def _forms_equal_signed(f1, f2, s):
    if len(f1) != len(f2):
        return False
    for k, v in f1.items():
        if f2.get(k, None) != s * v:
            return False
    return True

# ============================================================
# Triplet extraction (shared by equivariance check and conjugation)
# ============================================================

def _extract_triplets(M, off, n):
    """One pass over the variable part of M -> numpy arrays (I, J, ranks,
    vals) with a dense variable indexing, plus identity-part triplets and
    the rank -> key list. ranks are dense ints so numpy can sort them."""
    Is, Js, Ks, Vals = [], [], [], []
    a0I, a0J, a0V = [], [], []
    for i in range(len(M)):
        row = M[i]
        for j in range(len(M)):
            for k, val in row[j].items():
                fv = Fraction(val)
                assert fv.denominator == 1, "non-integer form coefficient"
                if k == 0:
                    a0I.append(i); a0J.append(j); a0V.append(int(fv))
                else:
                    Is.append(i); Js.append(j); Ks.append(k)
                    Vals.append(int(fv))
    var_keys = sorted(set(Ks))
    rank = {k: r for r, k in enumerate(var_keys)}
    I = np.asarray(Is, dtype=np.int64)
    J = np.asarray(Js, dtype=np.int64)
    R = np.fromiter((rank[k] for k in Ks), dtype=np.int64, count=len(Ks))
    V = np.asarray(Vals, dtype=np.int64)
    A0 = (np.asarray(a0I, dtype=np.int64), np.asarray(a0J, dtype=np.int64),
          np.asarray(a0V, dtype=np.int64))
    return I, J, R, V, A0, var_keys

# ============================================================
# Equivariance check (vectorized, exact)
# ============================================================

def _check_equivariance_fast(I, J, R, V, A0, mats, off, M0row=None):
    """M[g i][g j] == s_i s_j M[i][j] for the three generators, as sorted
    triplet-array equality. Operates on pre-extracted triplets (indices
    include the identity offset). M0row: identity row forms for the off=1
    case, checked separately."""
    def view(a, b, c, d):
        order = np.lexsort((d, c, b, a))
        return a[order], b[order], c[order], d[order]

    base = view(I, J, R, V)
    nfull = int(max(I.max(initial=0), J.max(initial=0))) + 1
    for g in [(1, 0), (4, 0), (0, 1)]:
        cm = mats[g]
        perm = np.arange(nfull, dtype=np.int64)
        sign = np.ones(nfull, dtype=np.int64)
        for col, (row, s) in cm.items():
            perm[off + col] = off + row
            sign[off + col] = s
        img = view(perm[I], perm[J], R, V * sign[I] * sign[J])
        for x, y in zip(base, img):
            assert np.array_equal(x, y), f"M not equivariant under {g}"
    if M0row is not None:
        for g in [(1, 0), (4, 0), (0, 1)]:
            cm = mats[g]
            for j, form in enumerate(M0row):
                rj, sj = cm[j]
                assert _forms_equal_signed(M0row[rj], form, sj), \
                    f"identity row not equivariant under {g}"

# ============================================================
# Projectors -> integer adapted vectors (sparse)
# ============================================================

def _project_columns(mats, n, irrep):
    d4, z2 = irrep
    dim = _DIMS[d4]
    cols = []
    for col in range(n):
        vec = {}
        for pg in range(8):
            chi_pg = _char(d4, pg)
            if chi_pg == 0:
                continue
            for sf in (0, 1):
                chi = chi_pg * (-1 if (z2 == 1 and sf == 1) else 1)
                row, sgn = mats[(pg, sf)][col]
                vec[row] = vec.get(row, Fraction(0)) \
                           + Fraction(dim * chi * sgn, 16)
        vec = {r: x for r, x in vec.items() if x != 0}
        if vec:
            cols.append(vec)
    return cols

def _project_E_row(mats, n, z2):
    cols = []
    for col in range(n):
        vec = {}
        for pg in range(8):
            d00 = PG8[pg][0][0]
            if d00 == 0:
                continue
            for sf in (0, 1):
                chi = d00 * (-1 if (z2 == 1 and sf == 1) else 1)
                row, sgn = mats[(pg, sf)][col]
                vec[row] = vec.get(row, Fraction(0)) \
                           + Fraction(2 * chi * sgn, 16)
        vec = {r: x for r, x in vec.items() if x != 0}
        if vec:
            cols.append(vec)
    return cols

def _row_reduce_int(cols, n):
    basis, pivots = [], []
    for vec in cols:
        v = dict(vec)
        for bvec, p in zip(basis, pivots):
            if p in v and v[p] != 0:
                f = v[p] / bvec[p]
                for r, x in bvec.items():
                    v[r] = v.get(r, Fraction(0)) - f * x
                    if v[r] == 0:
                        del v[r]
        if v:
            basis.append(v)
            pivots.append(min(v))
    out = []
    for v in basis:
        den = 1
        for x in v.values():
            den = den * x.denominator // gcd(den, x.denominator)
        dense = [0] * n
        for r, x in v.items():
            dense[r] = int(x * den)
        out.append(dense)
    return out

def adapted_vectors(basis_keys, P, dedup_E=False, mats=None):
    """{irrep: integer vectors}. dedup_E=True: E contributes one partner
    copy (production); bookkeeping n_1d + 2*n_E == n asserted either way."""
    n = len(basis_keys)
    if mats is None:
        mats = rep_matrices(basis_keys, P)
    _check_representation(mats)
    out, n1d, nE = {}, 0, 0
    for irrep in IRREPS:
        d4, z2 = irrep
        if d4 == "E" and dedup_E:
            vs = _row_reduce_int(_project_E_row(mats, n, z2), n)
            nE += len(vs)
        else:
            vs = _row_reduce_int(_project_columns(mats, n, irrep), n)
            if d4 == "E":
                assert len(vs) % 2 == 0
                nE += len(vs) // 2
            else:
                n1d += len(vs)
        if vs:
            out[irrep] = vs
    assert n1d + 2 * nE == n, f"isotypic bookkeeping: {n1d} + 2*{nE} != {n}"
    return out

# ============================================================
# Exact reference split (tests)
# ============================================================

def _form_combo(forms, coeffs):
    out = {}
    for c, form in zip(coeffs, forms):
        if c == 0:
            continue
        for k, v in form.items():
            out[k] = out.get(k, 0) + c * v
    return {k: v for k, v in out.items() if v != 0}

def split_moment_matrix(M, basis_keys, P):
    n = len(M)
    for i in range(n):
        for j in range(i):
            assert M[i][j] == M[j][i], "M not symmetric as forms"
    vs = adapted_vectors(basis_keys, P, dedup_E=False)
    flat = [(irrep, v) for irrep, vlist in vs.items() for v in vlist]
    MV = [[_form_combo(M[i], v) for i in range(n)] for _, v in flat]
    def entry(a, b):
        return _form_combo(MV[b], flat[a][1])
    m = len(flat)
    for a in range(m):
        for b in range(m):
            if flat[a][0] != flat[b][0]:
                assert not entry(a, b), \
                    f"cross-irrep {flat[a][0]}x{flat[b][0]} nonzero"
    blocks = []
    for irrep in vs:
        idxs = [a for a in range(m) if flat[a][0] == irrep]
        blocks.append((irrep, [[entry(a, b) for b in idxs] for a in idxs]))
    return blocks

# ============================================================
# Production split
# ============================================================

def _rank_triplets(Ks, Is, Js, Vals, A0):
    var_keys = sorted(set(Ks))
    rank = {k: r for r, k in enumerate(var_keys)}
    I = np.asarray(Is, dtype=np.int64)
    J = np.asarray(Js, dtype=np.int64)
    R = np.fromiter((rank[k] for k in Ks), dtype=np.int64, count=len(Ks))
    V = np.asarray(Vals, dtype=np.int64)
    return I, J, R, V, A0, var_keys

def _lower_triplets(B):
    r, c = np.nonzero(B)
    keep = r >= c
    return (r[keep].tolist(), c[keep].tolist(),
            B[r[keep], c[keep]].astype(float).tolist())

def split_to_blockdata(M, basis_keys, P, has_identity=False,
                       check_equivariance=True, log=print,
                       chunk=512, triplets=None, id_row=None):
    """-> [(irrep, dim, a0_triplets, {var_key: triplets})], lower triangle,
    MOSEK symmat convention.

    Conjugation as two sparse-sparse products over ALL variables at once
    (T @ W, reindex, W^T @ .), exploiting that adapted vectors have <= 16
    nonzeros (orbit averages). int64 throughout; cross-irrep sub-blocks
    asserted zero per variable (numeric Schur verification). chunk bounds
    memory by processing variables in groups."""
    off = 1 if has_identity else 0
    n = len(basis_keys)
    N = n + off
    if M is not None:
        assert len(M) == N

    t0 = time.time()
    mats = rep_matrices(basis_keys, P)
    if triplets is None:
        I, J, R, V, A0, var_keys = _extract_triplets(M, off, n)
    else:
        Is, Js, Rs, Vals, A0raw, rank_key = triplets
        I = np.frombuffer(Is, dtype=np.int64).copy()
        J = np.frombuffer(Js, dtype=np.int64).copy()
        R = np.frombuffer(Rs, dtype=np.int64).copy()
        V = np.frombuffer(Vals, dtype=np.int64).copy()
        A0 = tuple(np.frombuffer(a, dtype=np.int64).copy()
                   if len(a) else np.zeros(0, dtype=np.int64)
                   for a in A0raw)
        var_keys = list(rank_key)
    m = len(var_keys)
    log(f"    extract (n={n}): {time.time() - t0:.1f}s, nnz {len(V)}, "
        f"vars {m}")

    if check_equivariance:
        t0 = time.time()
        _check_equivariance_fast(I, J, R, V, A0, mats, off,
                                 M0row=(id_row if id_row is not None
                                        else (M[0][off:] if off and M is not None
                                              else None)))
        log(f"    equivariance check: {time.time()-t0:.1f}s")

    t0 = time.time()
    vs = adapted_vectors(basis_keys, P, dedup_E=True, mats=mats)
    log(f"    adapted_vectors: {time.time()-t0:.1f}s")

    t0 = time.time()
    # ---- W: all adapted vectors as one sparse (N x N') int64 matrix,
    #      columns grouped by irrep; identity attached to (A1, even) ----
    col_slices = []          # (irrep, lo, hi)
    wr, wc, wv = [], [], []
    ncol = 0
    for irrep, vecs in vs.items():
        extra = 1 if (has_identity and irrep == ("A1", 0)) else 0
        lo = ncol
        if extra:
            wr.append(0); wc.append(ncol); wv.append(1)
            ncol += 1
        for v in vecs:
            for row, x in enumerate(v):
                if x:
                    wr.append(off + row); wc.append(ncol); wv.append(x)
            ncol += 1
        col_slices.append((irrep, lo, ncol))
    Np = ncol
    W = sp.csr_matrix((np.asarray(wv, dtype=np.int64),
                       (np.asarray(wr), np.asarray(wc))), shape=(N, Np))
    WT = W.T.tocsr()

    # ---- sort triplets by variable; A0 appended as one extra "variable" ----
    order = np.argsort(R, kind="stable")
    I, J, V, R = I[order], J[order], V[order], R[order]
    if len(A0[0]):
        I = np.concatenate([I, A0[0]])
        J = np.concatenate([J, A0[1]])
        V = np.concatenate([V, A0[2]])
        R = np.concatenate([R, np.full(len(A0[0]), m, dtype=np.int64)])
    mtot = m + (1 if len(A0[0]) else 0)

    results = [dict() for _ in range(mtot)]     # r -> {irrep: dense block}
    for base in range(0, mtot, chunk):
        top = min(base + chunk, mtot)
        sel = (R >= base) & (R < top)
        Rl = R[sel] - base
        T = sp.csr_matrix((V[sel], (Rl * N + I[sel], J[sel])),
                          shape=((top - base) * N, N))
        T2 = (T @ W).tocoo()                     # rows (r, i), cols a
        r_idx, i_idx = T2.row // N, T2.row % N
        M3 = sp.csr_matrix((T2.data, (i_idx, r_idx * Np + T2.col)),
                           shape=(N, (top - base) * Np))
        B = (WT @ M3).tocsc()                    # rows b, cols (r, a)
        assert abs(B.data).max(initial=0) < 2 ** 62
        for r in range(top - base):
            sub = B[:, r * Np:(r + 1) * Np].tocsr()
            entry = {}
            nnz_diag = 0
            for irrep, lo, hi in col_slices:
                blk = sub[lo:hi, lo:hi]
                nnz_diag += blk.nnz
                if blk.nnz:
                    bc = blk.tocoo()
                    keep = bc.row >= bc.col
                    entry[irrep] = (bc.row[keep].tolist(),
                                    bc.col[keep].tolist(),
                                    bc.data[keep].astype(float).tolist())
            assert nnz_diag == sub.nnz, \
                "cross-irrep entries nonzero: Schur verification failed"
            results[base + r] = entry
        del T, T2, M3, B
    log(f"    conjugation: {time.time()-t0:.1f}s")

    a0_blocks = results[m] if len(A0[0]) else {}
    blocks = []
    for irrep, lo, hi in col_slices:
        dim = hi - lo
        pv = {}
        for r, k in enumerate(var_keys):
            trip = results[r].get(irrep)
            if trip is not None:
                pv[k] = trip
        blocks.append((irrep, dim,
                       a0_blocks.get(irrep, ([], [], [])), pv))
    return blocks