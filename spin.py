"""
spin.py -- SU(2) structure: ward quotient (variable side) and spin-multiplet
adaptation of operator bases (block side).

Variable side: the generated wards <[S^-,[S^+,O]]> = 0 over neutral
variables encode the FULL singlet selection rule; we eliminate the system
exactly and substitute, so MOSEK sees no ward rows and only singlet
variables. Annihilation of every ward form under the substitution is
asserted (the arbiter for any sign/ordering error in the elimination).

Block side: ad S^+ is a sparse integer matrix on each basis (bases closed
by construction). Highest-weight vectors (ker ad S^+ per Sz sector) give
one representative per multiplet; Wigner-Eckart makes dropped m-copies'
blocks equal, so keeping hw only is exact. NOTE the ordering constraint:
cross-SPIN-sector entries vanish only AFTER the ward quotient, so pre-
quotient blocks may not be split by total spin -- symmetry.py therefore
merges spin sectors within each D4 irrep (the hw dedup is still the win).
Spinflip is excluded from the block group in the spin path: it maps hw to
lw within a multiplet; its content is absorbed by the dedup + quotient.
"""

import time
from fractions import Fraction
from math import gcd
import numpy as np
import scipy.sparse as sp

from algebra import (commutator, op_of, sz_charge, support_sites)
from physics import local_spin_charge

# ============================================================
# Ward quotient (variable side)
# ============================================================

def build_ward_quotient(ward_forms, var_keys, log=print):
    """Exact elimination of the ward equality system.

    ward_forms: list of {key: int}, homogeneous (no constant term).
    var_keys:   sorted list of all neutral variable keys.
    Returns (kept_keys, subst): subst[key] = {kept_key: Fraction} expresses
    every original EV in quotient variables; kept keys map to themselves.
    Pivot = max key per row, so kept variables are lexicographically small
    (stable across runs)."""
    t0 = time.time()
    rows = []
    for f in ward_forms:
        assert 0 not in f, "ward form has a constant term"
        rows.append({k: Fraction(v) for k, v in f.items()})

    # forward elimination
    pivot_of = {}                 # pivot key -> row normalized to pivot 1
    for row in rows:
        r = dict(row)
        while r:
            p = max(r)
            if p in pivot_of:
                c = r[p]
                for k, v in pivot_of[p].items():
                    r[k] = r.get(k, Fraction(0)) - c * v
                    if r[k] == 0:
                        del r[k]
            else:
                c = r[p]
                pivot_of[p] = {k: v / c for k, v in r.items()}
                r = {}

    # back-substitution, increasing pivot order: forward elimination chose
    # pivot = max(row), so any other pivot key in a row is SMALLER than the
    # row's own pivot; resolving small pivots first means each substitution
    # splices in an already-resolved row, one pass suffices. pop(k, None):
    # an earlier substitution can cancel a later snapshot key.
    for p in sorted(pivot_of):
        r = pivot_of[p]
        for k in [k for k in list(r) if k != p and k in pivot_of]:
            c = r.pop(k, None)
            if c is None:
                continue
            for k2, v2 in pivot_of[k].items():
                if k2 == k:
                    continue
                r[k2] = r.get(k2, Fraction(0)) - c * v2
                if r[k2] == 0:
                    del r[k2]
        pivot_of[p] = r

    kept_keys = [k for k in var_keys if k not in pivot_of]
    kept_set = set(kept_keys)
    subst = {k: {k: Fraction(1)} for k in kept_keys}
    for p, r in pivot_of.items():
        # row reads: p + sum_{k != p} r[k]*k = 0  =>  p = -sum r[k]*k
        s = {k: -v for k, v in r.items() if k != p}
        assert all(k in kept_set for k in s), \
            "back-substitution left an unresolved pivot key"
        subst[p] = s
    log(f"ward quotient: {len(var_keys)} vars -> {len(kept_keys)} "
        f"({len(pivot_of)} eliminated), {time.time()-t0:.1f}s")

    # completeness self-check: every ward form vanishes under subst
    for f in ward_forms:
        acc = {}
        for k, v in f.items():
            for k2, w in subst[k].items():
                acc[k2] = acc.get(k2, Fraction(0)) + v * w
        assert all(x == 0 for x in acc.values()), \
            "substitution fails to annihilate a ward form"
    return kept_keys, subst

# ============================================================
# Multiplet adaptation (block side)
# ============================================================

def _ad_Sp_matrix(basis_keys):
    """Sparse integer matrix of ad S^+ on span(basis_keys). Localizes to
    each monomial's support (disjoint even terms cancel exactly);
    closure asserted."""
    index = {k: i for i, k in enumerate(basis_keys)}
    rows, cols, vals = [], [], []
    for col, k in enumerate(basis_keys):
        img = commutator(local_spin_charge(support_sites(k), +1), op_of(k))
        for k2, v in img.items():
            assert k2 in index, "basis not closed under ad S^+"
            rows.append(index[k2]); cols.append(col); vals.append(v)
    n = len(basis_keys)
    return sp.csr_matrix((np.asarray(vals, dtype=np.int64),
                          (np.asarray(rows), np.asarray(cols))),
                         shape=(n, n))

def _integer_kernel(A, ncols):
    """Exact kernel of a sparse integer matrix, as integer row vectors
    (in the column index space 0..ncols-1)."""
    if A is None or A.shape[0] == 0 or A.nnz == 0:
        return [[1 if a == b else 0 for a in range(ncols)]
                for b in range(ncols)]
    Ad = A.toarray()
    rows = [{j: Fraction(int(Ad[i, j])) for j in range(ncols)
             if Ad[i, j] != 0} for i in range(A.shape[0])]
    pivot_of = {}
    for row in rows:
        r = dict(row)
        while r:
            p = min(r)
            if p in pivot_of:
                c = r[p]
                for k, v in pivot_of[p].items():
                    r[k] = r.get(k, Fraction(0)) - c * v
                    if r[k] == 0:
                        del r[k]
            else:
                c = r[p]
                pivot_of[p] = {k: v / c for k, v in r.items()}
                r = {}
    free = [j for j in range(ncols) if j not in pivot_of]
    kern = []
    for fcol in free:
        v = {fcol: Fraction(1)}
        for p in sorted(pivot_of, reverse=True):
            row = pivot_of[p]
            s = sum(row.get(k, Fraction(0)) * v.get(k, Fraction(0))
                    for k in list(v) if k != p)
            if s != 0:
                v[p] = -s
        den = 1
        for x in v.values():
            den = den * x.denominator // gcd(den, x.denominator)
        kern.append([int(v.get(j, Fraction(0)) * den)
                     for j in range(ncols)])
    return kern

def highest_weight_vectors(basis_keys, log=print):
    """{two_s: integer vectors in the full basis index space} -- one hw
    vector per multiplet (spin s lives in the Sz = +s sector as
    ker ad S^+). Multiplet sum rule asserted:
    sum (two_s + 1) * count == len(basis_keys)."""
    t0 = time.time()
    n = len(basis_keys)
    Sp = _ad_Sp_matrix(basis_keys)
    by_m = {}
    for i, k in enumerate(basis_keys):
        by_m.setdefault(sz_charge(k), []).append(i)

    out, total = {}, 0
    for two_m in sorted(by_m, reverse=True):
        if two_m < 0:
            continue
        idxs = np.asarray(by_m[two_m], dtype=np.int64)
        target = by_m.get(two_m + 2, [])
        if target:
            sub = Sp[np.asarray(target, dtype=np.int64)][:, idxs]
            kern = _integer_kernel(sub, len(idxs))
        else:
            kern = [[1 if a == b else 0 for a in range(len(idxs))]
                    for b in range(len(idxs))]
        vecs = []
        for v in kern:
            full = [0] * n
            for a, i in enumerate(idxs):
                full[i] = v[a]
            vecs.append(full)
        if vecs:
            out[two_m] = vecs
            total += (two_m + 1) * len(vecs)
    assert total == n, f"multiplet sum rule: {total} != {n}"
    log(f"    hw extraction (n={n}): {time.time()-t0:.1f}s, "
        f"multiplets {{2s: count}} = "
        f"{ {s: len(v) for s, v in out.items()} }")
    return out

def spin_adapt_matrix(basis_keys, log=print):
    """Integer matrix U (n x n_kept) of hw multiplet representatives,
    with labels[a] = two_s of column a. Consumed by
    symmetry.split_to_blockdata(spin_U=U, spin_labels=labels)."""
    hw = highest_weight_vectors(basis_keys, log=log)
    n = len(basis_keys)
    cols, labels = [], []
    for two_s in sorted(hw):
        for v in hw[two_s]:
            cols.append(v)
            labels.append(two_s)
    U = np.zeros((n, len(cols)), dtype=np.int64)
    for a, v in enumerate(cols):
        U[:, a] = v
    return U, labels