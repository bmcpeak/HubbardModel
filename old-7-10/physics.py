"""
physics.py -- Hamiltonian, objective, adjoint action, Ward identities,
G blocks, eigenstate constraints, charge pruning. Everything here has an
independent certificate: ad_H and wards via test_oracle.py, the H-linear
constraint path via the Mathematica cross-stack verification.

Coordinate discipline (see algebra.py docstring): ad_H asserts interior
input; callers shift by (OX, OY) via shift_op, and both factors of any
product must share the offset.
"""

from functools import lru_cache
from fractions import Fraction

from algebra import (add, multiply, commutator, dagger, c_op, d_op, op_of,
                     unpack, sites_of, support_sites, shift_op, shift_key,
                     to_linear_form, normalize_form, canon,
                     n_charge, sz_charge, show, UP, DN)

# Interior anchor for support-growing operations (ad_H adds one ring).
# Invariant: (P - 1) + OX + 1 < W for the active patch; run.py asserts it.
OX, OY = 2, 2

# ============================================================
# Objective and filling
# ============================================================

def hop_part():
    h = {}
    for p, q in [((0, 0), (0, 1)), ((0, 0), (1, 0))]:
        for s in (DN, UP):
            h = add(h, multiply(d_op(*p, s), c_op(*q, s)), scale=-1)
            h = add(h, multiply(d_op(*q, s), c_op(*p, s)), scale=-1)
    return h

def u_part():
    return multiply(multiply(d_op(0, 0, UP), c_op(0, 0, UP)),
                    multiply(d_op(0, 0, DN), c_op(0, 0, DN)))

def objective(t, U):
    out = {}
    for k, v in to_linear_form(hop_part()).items():
        out[k] = out.get(k, 0) + t * v
    for k, v in to_linear_form(u_part()).items():
        out[k] = out.get(k, 0) + U * v
    return out

def filling_constraint(nu):
    n0 = add(multiply(d_op(0, 0, UP), c_op(0, 0, UP)),
             multiply(d_op(0, 0, DN), c_op(0, 0, DN)))
    return to_linear_form(n0), Fraction(nu)

# ============================================================
# Adjoint action of H
# ============================================================

def _hop_terms_touching(sites):
    bonds = set()
    for (i, j) in sites:
        for (pi, pj) in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
            bonds.add(frozenset([(i, j), (pi, pj)]))
    terms = {}
    for bond in bonds:
        p, q = tuple(bond)
        for s in (DN, UP):
            terms = add(terms, multiply(d_op(*p, s), c_op(*q, s)))
            terms = add(terms, multiply(d_op(*q, s), c_op(*p, s)))
    return terms

def _u_terms_touching(sites):
    out = {}
    for (i, j) in sites:
        out = add(out, multiply(multiply(d_op(i, j, UP), c_op(i, j, UP)),
                                multiply(d_op(i, j, DN), c_op(i, j, DN))))
    return out

@lru_cache(maxsize=None)
def ad_H(key):
    """[H, monomial] as (t_items, U_items); t_items multiply -t downstream.
    Input must be interior (caller shifts); output in SAME coordinates."""
    sites = support_sites(key)
    assert all(i >= 1 and j >= 1 for i, j in sites), \
        "ad_H on edge-anchored key: shift_op first"
    O = op_of(key)
    return (tuple(commutator(_hop_terms_touching(sites), O).items()),
            tuple(commutator(_u_terms_touching(sites), O).items()))

def ad_H_op(op, t, U):
    out = {}
    for key, v in op.items():
        tp, up = ad_H(key)
        for k, w in tp:
            out[k] = out.get(k, 0) - t * v * w
        for k, w in up:
            out[k] = out.get(k, 0) + U * v * w
    return {k: v for k, v in out.items() if v != 0}

# ============================================================
# Eigenstate equalities <[H, O]> = 0
# ============================================================

def linear_H_constraints(variables, span, t, U):
    """In-span forms only (out-of-span would need new variables; dropping
    them is exact, unlike truncating). Verified cross-stack in Mathematica."""
    kept, out_of_span = [], 0
    seen = set()
    for v in sorted(variables):
        form = to_linear_form(ad_H_op(shift_op(op_of(v), OX, OY), t, U))
        if not form:
            continue
        if set(form) <= span:
            h = normalize_form(form)
            if h not in seen:
                seen.add(h)
                kept.append(form)
        else:
            out_of_span += 1
    return kept, out_of_span

# ============================================================
# Ward identities (singlet assumption)
# ============================================================

def local_spin_charge(sites, pm):
    S = {}
    for i, j in sites:
        if pm > 0:
            S = add(S, multiply(d_op(i, j, UP), c_op(i, j, DN)))
        else:
            S = add(S, multiply(d_op(i, j, DN), c_op(i, j, UP)))
    return S

def ward_constraints(neutral_vars):
    span = set(neutral_vars) | {0}
    seen, out = set(), []
    dropped = 0
    for v in sorted(neutral_vars):
        raised = commutator(local_spin_charge(support_sites(v), +1), op_of(v))
        for mu in sorted(raised):
            assert sz_charge(mu) == 2
            form = to_linear_form(
                commutator(local_spin_charge(support_sites(mu), -1), op_of(mu)))
            if not form:
                continue
            if not set(form) <= span:
                dropped += 1
                continue
            h = normalize_form(form)
            if h not in seen:
                seen.add(h)
                out.append(form)
    return out, dropped

# ============================================================
# G blocks
# ============================================================

def build_G_block(cands, span, t, U, log=print):
    for o in cands:
        for k in o:
            assert n_charge(k) == 0, "number-changing G candidate"
    scands = [shift_op(o, OX, OY) for o in cands]
    hops = [ad_H_op(o, t, U) for o in scands]
    dags = [dagger(o) for o in scands]

    pre = [i for i in range(len(scands))
           if set(to_linear_form(multiply(dags[i], hops[i]))) <= span]
    log(f"  G prefilter: {len(pre)}/{len(cands)} pass the diagonal test")
    if not pre:
        return [], [], []

    n = len(pre)
    form = [[to_linear_form(multiply(dags[pre[i]], hops[pre[j]]))
             for j in range(n)] for i in range(n)]
    ok = [[set(form[i][j]) <= span for j in range(n)] for i in range(n)]

    alive = set(range(n))
    while alive:
        viol = {i: sum((not ok[i][j]) + (not ok[j][i]) for j in alive)
                for i in alive}
        worst = max(viol, key=lambda i: viol[i])
        if viol[worst] == 0:
            break
        alive.discard(worst)
    keep_local = sorted(alive)
    keep = [pre[i] for i in keep_local]
    G = [[form[i][j] for j in keep_local] for i in keep_local]

    harvested, seen = [], set()
    m = len(keep_local)
    for i in range(m):
        for j in range(i):
            keys = set(G[i][j]) | set(G[j][i])
            anti = {k: v for k in keys
                    if (v := G[i][j].get(k, 0) - G[j][i].get(k, 0)) != 0}
            sym = {k: v for k in keys
                   if (v := Fraction(G[i][j].get(k, 0)
                                     + G[j][i].get(k, 0), 2)) != 0}
            G[i][j] = dict(sym)
            G[j][i] = dict(sym)
            if anti:
                h = normalize_form(anti)
                if h not in seen:
                    seen.add(h)
                    harvested.append(anti)
    return keep, G, harvested

def sanity_check_G():
    doub = shift_op(u_part(), OX, OY)
    entry = to_linear_form(multiply(dagger(doub), ad_H_op(doub, 1, 8)))
    assert entry, "G doublon diagonal is zero: coordinate bug is back"
    return entry

# ============================================================
# Pruning and scrubbing (Sz selection rule)
# ============================================================

def prune_charged(mats):
    for m in mats:
        for row in m:
            for entry in row:
                for k in [k for k in entry if k != 0 and sz_charge(k) != 0]:
                    del entry[k]

def scrub_forms(forms):
    out = []
    for f in forms:
        f2 = {k: v for k, v in f.items() if k == 0 or sz_charge(k) == 0}
        if f2:
            out.append(f2)
    return out