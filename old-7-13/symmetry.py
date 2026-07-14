"""
symmetry.py -- point-group irrep decomposition of operator bases.

Group: G = D4 x Z2(spinflip), |G| = 16, acting on monomial bases as signed
permutations. Dagger excluded (maps charge q to -q; lives in canon).

Paths:
  split_moment_matrix  -- exact Fraction reference (pytest; unchanged)
  split_to_blockdata   -- production: triplet intake, sparse two-product
                          conjugation, triplet output, A1-attached
                          identity, E-dedup, per-variable Schur check.

Spin path (spin_U given): adapted vectors are D4-ONLY projections of the
highest-weight multiplet representatives from spin.py. Spinflip is absent
by design (it maps hw to lw within multiplets; its content is absorbed by
the multiplet dedup + ward quotient). Spin sectors are MERGED within each
D4 irrep: cross-spin entries vanish only after the ward quotient, which
runs downstream, so pre-quotient blocks may not be split by total spin.
Block keys are plain d4 names ("A1", ...) in this path, (d4, z2) tuples in
the standard path; _is_trivial handles the identity attachment for both.
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

def _is_trivial(irrep):
    """Identity-attachment test for both key shapes."""
    return irrep == ("A1", 0) or irrep == "A1"

# ============================================================
# Group action on a monomial basis
# ============================================================

def _act_centered(pg_index, sf, key, P):
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
    def comp(m1, m2):
        return {c: (m1[m][0], m1[m][1] * s2) for c, (m, s2) in m2.items()}
    ident = {c: (c, 1) for c in range(len(basis_keys))}
    r90, mj, sfm = mats[(1, 0)], mats[(4, 0)], mats[(0, 1)]
    assert comp(r90, comp(r90, comp(r90, r90))) == ident, "r90^4 != e"
    assert comp(mj, mj) == ident, "mj^2 != e"
    assert comp(sfm, sfm) == ident, "sf^2 != e"
    assert comp(r90, r90) == mats[(2, 0)], "r90^2 != r180"
    return mats

def _orbit_representatives(mats, n):
    seen = bytearray(n)
    reps = []
    for col in range(n):
        if seen[col]:
            continue
        reps.append(col)
        for cm in mats.values():
            seen[cm[col][0]] = 1
    return reps

def _signed_trace(cm):
    return sum(s for c, (r, s) in cm.items() if r == c)

def _multiplicities(mats, n):
    traces = {g: _signed_trace(cm) for g, cm in mats.items()}
    out = {}
    for d4 in _D4_CHARS:
        for z2 in (0, 1):
            acc = 0
            for pg in range(8):
                for sf in (0, 1):
                    chi = _char(d4, pg) * (-1 if (z2 == 1 and sf == 1) else 1)
                    acc += chi * traces[(pg, sf)]
            assert acc % 16 == 0, "non-integer multiplicity: rep broken"
            out[(d4, z2)] = acc // 16
    return out

# ============================================================
# Projectors -> integer adapted vectors
# ============================================================

def _project_col(mats, col, weight):
    vec = {}
    for pg in range(8):
        for sf in (0, 1):
            w = weight(pg, sf)
            if w == 0:
                continue
            row, sgn = mats[(pg, sf)][col]
            vec[row] = vec.get(row, Fraction(0)) + w * sgn
    return {r: x for r, x in vec.items() if x != 0}

def _reduce_lazy(col_iter, expected):
    """Sparse Fraction row-reduction; stops at `expected` vectors if given,
    else exhausts the iterator."""
    basis, pivots = [], []
    for vec in col_iter:
        if not vec:
            continue
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
            if expected is not None and len(basis) == expected:
                break
    return basis

def _integerize(basis, n):
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
    """{(d4, z2): integer vectors} over D4 x Z2. dedup_E=True: one E
    partner per multiplet. Bookkeeping asserted."""
    n = len(basis_keys)
    if mats is None:
        mats = rep_matrices(basis_keys, P)
    mult = _multiplicities(mats, n)
    reps = _orbit_representatives(mats, n)
    rep_set = set(reps)
    col_order = reps + [c for c in range(n) if c not in rep_set]

    out, n1d, nE = {}, 0, 0
    for irrep in IRREPS:
        d4, z2 = irrep
        m = mult[irrep]
        if m == 0:
            continue
        if d4 == "E" and dedup_E:
            def w(pg, sf, z2=z2):
                d00 = PG8[pg][0][0]
                return Fraction(0) if d00 == 0 else \
                    Fraction(2 * d00 * (-1 if (z2 == 1 and sf == 1) else 1),
                             16)
            expected = m
        else:
            def w(pg, sf, d4=d4, z2=z2):
                chi = _char(d4, pg)
                return Fraction(0) if chi == 0 else \
                    Fraction(_DIMS[d4] * chi
                             * (-1 if (z2 == 1 and sf == 1) else 1), 16)
            expected = _DIMS[d4] * m
        basis = _reduce_lazy(
            (_project_col(mats, c, w) for c in col_order), expected)
        assert len(basis) == expected, \
            f"{irrep}: found {len(basis)} of {expected}"
        out[irrep] = _integerize(basis, n)
        if d4 == "E":
            nE += m
        else:
            n1d += expected
    assert n1d + 2 * nE == n, f"isotypic bookkeeping: {n1d} + 2*{nE} != {n}"
    return out

# ---- spin path: D4-only projection of hw representatives, merged 2s ----

def _project_col_vec(d4mats, colvec, weight):
    """D4-only (8-element) projector applied to an integer vector given as
    {row: int}. weight(pg) -> Fraction."""
    out = {}
    for pg in range(8):
        w = weight(pg)
        if w == 0:
            continue
        cm = d4mats[pg]
        for r, x in colvec.items():
            row, sgn = cm[r]
            out[row] = out.get(row, Fraction(0)) + w * sgn * x
    return {r: v for r, v in out.items() if v != 0}

def adapted_vectors_spin(basis_keys, P, spin_U, spin_labels,
                         mats=None, log=print):
    """{d4name: integer vectors}: hw multiplet representatives refined by
    D4-only isotypic projection, spin sectors MERGED per d4 irrep (see
    module docstring for why). E-dedup via the E-row projector.
    Bookkeeping per merged irrep set: n1d + 2*nE == n_hw columns."""
    n = len(basis_keys)
    if mats is None:
        mats = rep_matrices(basis_keys, P)
    d4mats = {pg: mats[(pg, 0)] for pg in range(8)}

    cols = []
    for a in range(spin_U.shape[1]):
        rows = np.nonzero(spin_U[:, a])[0]
        cols.append({int(r): int(spin_U[r, a]) for r in rows})
    n_hw = len(cols)

    out, n1d, nE = {}, 0, 0
    for d4name in _D4_CHARS:
        if d4name == "E":
            def w(pg):
                d00 = PG8[pg][0][0]
                return Fraction(2 * d00, 8) if d00 else Fraction(0)
        else:
            def w(pg, d4name=d4name):
                chi = _char(d4name, pg)
                return Fraction(chi, 8) if chi else Fraction(0)
        basis = _reduce_lazy(
            (_project_col_vec(d4mats, c, w) for c in cols), expected=None)
        if basis:
            out[d4name] = _integerize(basis, n)
            if d4name == "E":
                nE += len(basis)
            else:
                n1d += len(basis)
    assert n1d + 2 * nE == n_hw, \
        f"spin-path bookkeeping: {n1d} + 2*{nE} != {n_hw}"
    log(f"    spin+D4 adaptation: {n_hw} hw cols -> "
        f"{ {k: len(v) for k, v in out.items()} }")
    return out

# ============================================================
# Equivariance check (vectorized, exact)
# ============================================================

def _forms_equal_signed(f1, f2, s):
    if len(f1) != len(f2):
        return False
    for k, v in f1.items():
        if f2.get(k, None) != s * v:
            return False
    return True

def _check_equivariance_fast(I, J, R, V, A0, mats, off, M0row=None):
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
# Exact reference split (pytest; unchanged)
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
# Production split (triplets in, triplets out)
# ============================================================

def split_to_blockdata(M, basis_keys, P, has_identity=False,
                       check_equivariance=True, log=print,
                       chunk=256, triplets=None, id_row=None, mats=None,
                       spin_U=None, spin_labels=None):
    """-> [(irrep, dim, a0_triplets, {var_key: triplets})], lower triangle.
    Never materializes dense per-variable blocks. Cross-irrep vanishing
    asserted per variable. With spin_U: hw-deduped, D4-only, spin sectors
    merged per irrep (keys are d4 name strings)."""
    off = 1 if has_identity else 0
    n = len(basis_keys)
    N = n + off
    if M is not None:
        assert len(M) == N

    t0 = time.time()
    if mats is None:
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
    log(f"    intake (n={n}): {time.time()-t0:.1f}s, nnz {len(V)}, vars {m}")

    if check_equivariance:
        t0 = time.time()
        _check_equivariance_fast(I, J, R, V, A0, mats, off,
                                 M0row=id_row if id_row is not None
                                 else (M[0][off:] if off and M is not None
                                       else None))
        log(f"    equivariance check: {time.time()-t0:.1f}s")

    t0 = time.time()
    if spin_U is not None:
        vs = adapted_vectors_spin(basis_keys, P, spin_U, spin_labels,
                                  mats=mats, log=log)
    else:
        vs = adapted_vectors(basis_keys, P, dedup_E=True, mats=mats)
    log(f"    adapted_vectors: {time.time()-t0:.1f}s")

    t0 = time.time()
    col_slices, wr, wc, wv, ncol = [], [], [], [], 0
    for irrep, vecs in vs.items():
        extra = 1 if (has_identity and _is_trivial(irrep)) else 0
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
    if has_identity:
        assert any(_is_trivial(ir) for ir, _, _ in col_slices), \
            "identity requires a trivial-irrep component"
    W = sp.csr_matrix((np.asarray(wv, dtype=np.int64),
                       (np.asarray(wr), np.asarray(wc))), shape=(N, Np))
    WT = W.T.tocsr()

    order = np.argsort(R, kind="stable")
    I, J, V, R = I[order], J[order], V[order], R[order]
    if len(A0[0]):
        I = np.concatenate([I, A0[0]])
        J = np.concatenate([J, A0[1]])
        V = np.concatenate([V, A0[2]])
        R = np.concatenate([R, np.full(len(A0[0]), m, dtype=np.int64)])
    mtot = m + (1 if len(A0[0]) else 0)

    results = [None] * mtot
    for base in range(0, mtot, chunk):
        top = min(base + chunk, mtot)
        sel = (R >= base) & (R < top)
        Rl = R[sel] - base
        T = sp.csr_matrix((V[sel], (Rl * N + I[sel], J[sel])),
                          shape=((top - base) * N, N))
        T2 = (T @ W).tocoo()
        r_idx, i_idx = T2.row // N, T2.row % N
        M3 = sp.csr_matrix((T2.data, (i_idx, r_idx * Np + T2.col)),
                           shape=(N, (top - base) * Np))
        B = (WT @ M3).tocsc()
        B.eliminate_zeros()
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
            trip = results[r].get(irrep) if results[r] else None
            if trip is not None:
                pv[k] = trip
        blocks.append((irrep, dim,
                       a0_blocks.get(irrep, ([], [], [])), pv))
    return blocks

# ---- form-matrix extraction (unsplit / reference path) ----

def _extract_triplets(M, off, n):
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