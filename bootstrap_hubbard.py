"""
Infinite-volume Hubbard bootstrap.

  M blocks:  <O^dag O>      >= 0   (moment matrices)
  G blocks:  <O^dag [H,O]>  >= 0   (ground-state matrices; number-conserving
                                    O only -- validity at fixed filling)
  Equalities: filling, SU(2) Ward identities, harvested eigenstate
              constraints <[H,O]> = 0 where in-span.

All bootstrap content -- which operators enter M and G -- lives in
Section 6 (CONFIGURATION) and nowhere else.

Coordinate discipline (source of the last bug, read this):
  canon anchors everything at (0,0). ad_H grows support by one ring, so any
  operator fed to ad_H must first be shifted into the window interior by
  (OX, OY) -- and CRUCIALLY, when forming products like O^dag [H, O], BOTH
  factors must live at the same offset. canon erases a global shift, not a
  relative one. ad_H asserts its input is interior; shift_op is the tool.

Representation:
  - Monomial = packed int (D << NMODES) | C over window modes.
    d's ascending, c's descending; dagger = sign-free swap (D,C)->(C,D).
  - Operator = dict {key: coeff}, normal-ordered monomials only.
  - Linear form = dict {canonical_key: coeff}; key 0 = constant term.

Physics assumptions:
  - Reality <O^dag> = <O> (dagger in symmetry group).
  - Singlet ground state (Ward identities; Sz selection rule).
  - G blocks and <[H,O]> = 0 valid in the ground state.
"""

import sys
import time
from itertools import combinations
from functools import lru_cache
from fractions import Fraction

def log(msg):
    print(msg, flush=True)

# ============================================================
# 1. Window and modes
# ============================================================

W = 10
NMODES = 2 * W * W
CMASK = (1 << NMODES) - 1
UP, DN = 1, 0

def mode(i, j, s):
    assert 0 <= i < W and 0 <= j < W, f"site ({i},{j}) outside window"
    return 2 * (W * i + j) + s

def site_of_mode(m):
    s = m & 1
    m >>= 1
    return (m // W, m % W, s)

# ============================================================
# 2. Packing
# ============================================================

def pack(D, C):
    return (D << NMODES) | C

def unpack(key):
    return key >> NMODES, key & CMASK

def sites_of(mask):
    out = []
    while mask:
        m = (mask & -mask).bit_length() - 1
        out.append(site_of_mode(m))
        mask &= mask - 1
    return out

def pack_sites(Dsites, Csites):
    D = C = 0
    for p in Dsites: D |= 1 << mode(*p)
    for p in Csites: C |= 1 << mode(*p)
    return pack(D, C)

def shift_key(key, di, dj):
    """Uniform translation: order-preserving, hence sign-free."""
    D, C = unpack(key)
    Ds = [(i + di, j + dj, s) for i, j, s in sites_of(D)]
    Cs = [(i + di, j + dj, s) for i, j, s in sites_of(C)]
    return pack_sites(Ds, Cs)

def shift_op(op, di, dj):
    return {shift_key(k, di, dj): v for k, v in op.items()}

# ============================================================
# 3. Algebra
# ============================================================

def mul_c(terms, m):
    out = {}
    bit = 1 << m
    below = bit - 1
    for key, coeff in terms.items():
        D, C = key >> NMODES, key & CMASK
        if C & bit:
            continue
        sign = -1 if (C & below).bit_count() & 1 else 1
        k = (D << NMODES) | (C | bit)
        out[k] = out.get(k, 0) + sign * coeff
    return out

def mul_d(terms, m):
    out = {}
    bit = 1 << m
    below = bit - 1
    for key, coeff in terms.items():
        D, C = key >> NMODES, key & CMASK
        if C & bit:
            s = -1 if (C & below).bit_count() & 1 else 1
            k = (D << NMODES) | (C ^ bit)
            out[k] = out.get(k, 0) + s * coeff
        if not (D & bit):
            p = C.bit_count() + (D >> (m + 1)).bit_count()
            s = -1 if p & 1 else 1
            k = ((D | bit) << NMODES) | C
            out[k] = out.get(k, 0) + s * coeff
    return out

def factors(key):
    D, C = unpack(key)
    seq = []
    while D:
        m = (D & -D).bit_length() - 1
        seq.append(('d', m)); D &= D - 1
    cs = []
    while C:
        m = (C & -C).bit_length() - 1
        cs.append(('c', m)); C &= C - 1
    seq.extend(reversed(cs))
    return seq

@lru_cache(maxsize=None)
def _mono_product(ka, kb):
    term = {ka: 1}
    for kind, m in factors(kb):
        term = mul_d(term, m) if kind == 'd' else mul_c(term, m)
    return tuple(term.items())

def multiply(A, B):
    out = {}
    for kb, vb in B.items():
        for ka, va in A.items():
            for k, v in _mono_product(ka, kb):
                out[k] = out.get(k, 0) + va * vb * v
    return {k: v for k, v in out.items() if v != 0}

def add(A, B, scale=1):
    out = dict(A)
    for k, v in B.items():
        out[k] = out.get(k, 0) + scale * v
        if out[k] == 0:
            del out[k]
    return out

def commutator(A, B):
    return add(multiply(A, B), multiply(B, A), scale=-1)

def dagger(op):
    out = {}
    for key, v in op.items():
        D, C = unpack(key)
        out[pack(C, D)] = v
    return out

def c_op(i, j, s): return {pack(0, 1 << mode(i, j, s)): 1}
def d_op(i, j, s): return {pack(1 << mode(i, j, s), 0): 1}

IDENTITY = {0: 1}

def op_of(key):
    return {key: 1}

# ============================================================
# 4. Symmetry group and canonicalization
# ============================================================

PG8 = [
    (( 1, 0), ( 0, 1)), (( 0,-1), ( 1, 0)), ((-1, 0), ( 0,-1)), (( 0, 1), (-1, 0)),
    (( 1, 0), ( 0,-1)), ((-1, 0), ( 0, 1)), (( 0, 1), ( 1, 0)), (( 0,-1), (-1, 0)),
]
FINITE = [(M, sf, dag) for M in PG8 for sf in (0, 1) for dag in (0, 1)]

def act(M, sf, p):
    (a, b), (c_, d_) = M
    i, j, s = p
    return (a * i + b * j, c_ * i + d_ * j, s ^ sf)

def parity_sign(sites):
    seq = [mode(*p) for p in sites]
    inv = sum(1 for a in range(len(seq)) for b in range(a + 1, len(seq))
              if seq[a] > seq[b])
    return -1 if inv & 1 else 1

def _images(key):
    Dsites, Csites = sites_of(key >> NMODES), sites_of(key & CMASK)
    out = []
    for M, sf, dag in FINITE:
        Ds, Cs = (Csites, Dsites) if dag else (Dsites, Csites)
        Ds = [act(M, sf, p) for p in Ds]
        Cs = [act(M, sf, p) for p in Cs]
        di = min(p[0] for p in Ds + Cs)
        dj = min(p[1] for p in Ds + Cs)
        Ds = [(i - di, j - dj, s) for i, j, s in Ds]
        Cs = [(i - di, j - dj, s) for i, j, s in Cs]
        out.append((pack_sites(Ds, Cs), parity_sign(Ds) * parity_sign(Cs)))
    return out

@lru_cache(maxsize=None)
def canon(key):
    if key == 0:
        return (0, 1)
    images = _images(key)
    kmin = min(k for k, _ in images)
    signs = {s for k, s in images if k == kmin}
    return (kmin, 0) if len(signs) == 2 else (kmin, signs.pop())

def to_linear_form(op):
    form = {}
    for key, v in op.items():
        k, s = canon(key)
        if s == 0:
            continue
        form[k] = form.get(k, 0) + s * v
        if form[k] == 0:
            del form[k]
    return form

def _normalize(form):
    kmin = min(form)
    lead = Fraction(form[kmin])
    return tuple(sorted((k, Fraction(v) / lead) for k, v in form.items()))

# ============================================================
# 5. Printing
# ============================================================

def show(key):
    if key == 0:
        return "1"
    Dsites, Csites = sites_of(key >> NMODES), sites_of(key & CMASK)
    parts  = [f"d[{i},{j},{s}]" for i, j, s in Dsites]
    parts += [f"c[{i},{j},{s}]" for i, j, s in reversed(Csites)]
    return "**".join(parts)

def show_op(op):
    if not op:
        return "0"
    return "  +  ".join(f"({v}) {show(k)}" for k, v in sorted(op.items()))

# ============================================================
# 6. CONFIGURATION -- all bootstrap content is chosen HERE
# ============================================================

PATCH = [(i, j) for i in range(3) for j in range(3)]

# Interior anchor for ad_H (adds one ring of support).
# Invariant: max candidate coordinate + OX + 1 < W.
OX, OY = 2, 2

def build_M_bases():
    b1 = [c_op(i, j, DN) for i, j in PATCH]
    b2 = [c_op(i, j, UP) for i, j in PATCH]
    b3 = [d_op(i, j, DN) for i, j in PATCH]
    b4 = [d_op(i, j, UP) for i, j in PATCH]
    b5  = [multiply(c_op(*p, DN), c_op(*q, DN)) for p, q in combinations(PATCH, 2)]
    b6  = [multiply(c_op(*p, UP), c_op(*q, DN)) for p in PATCH for q in PATCH]
    b7  = [multiply(c_op(*p, UP), c_op(*q, UP)) for p, q in combinations(PATCH, 2)]
    b8  = [multiply(d_op(*p, DN), d_op(*q, DN)) for p, q in combinations(PATCH, 2)]
    b9  = [multiply(d_op(*p, UP), d_op(*q, DN)) for p in PATCH for q in PATCH]
    b10 = [multiply(d_op(*p, UP), d_op(*q, UP)) for p, q in combinations(PATCH, 2)]
    b11 = [multiply(d_op(*p, 1 - k), c_op(*q, k))
           for p in PATCH for q in PATCH for k in (DN, UP)]
    b12 = [IDENTITY] + [multiply(d_op(*p, k), c_op(*q, k))
                        for p in PATCH for q in PATCH for k in (DN, UP)]
    return [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12]

def build_G_candidates():
    """Number-conserving only. The in-span filter removes what doesn't fit."""
    cands = [multiply(d_op(*p, a), c_op(*q, b))
             for p in PATCH for q in PATCH for a in (DN, UP) for b in (DN, UP)]
    cands += [multiply(multiply(d_op(*p, UP), c_op(*p, UP)),
                       multiply(d_op(*p, DN), c_op(*p, DN))) for p in PATCH]
    return cands

# ============================================================
# 7. Charges, pruning
# ============================================================

def n_charge(key):
    D, C = unpack(key)
    return D.bit_count() - C.bit_count()

def sz_charge(key):
    D, C = unpack(key)
    q = sum(1 if s == UP else -1 for _, _, s in sites_of(D))
    q -= sum(1 if s == UP else -1 for _, _, s in sites_of(C))
    return q

def collect_variables(mats):
    vs = set()
    for m in mats:
        for row in m:
            for entry in row:
                vs.update(entry.keys())
    vs.discard(0)
    return vs

def split_variables(mats):
    vs = collect_variables(mats)
    for v in vs:
        assert n_charge(v) == 0, f"N-charged variable: {show(v)}"
    neutral = {v for v in vs if sz_charge(v) == 0}
    return neutral, vs - neutral

def prune_charged(mats):
    for m in mats:
        for row in m:
            for entry in row:
                for k in [k for k in entry if k != 0 and sz_charge(k) != 0]:
                    del entry[k]

# ============================================================
# 8. Hamiltonian: objective, adjoint action, eigenstate constraints
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

def support_sites(key):
    D, C = unpack(key)
    return {(i, j) for i, j, s in sites_of(D | C)}

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
    Input must already be interior (caller shifts); output is in the SAME
    coordinates as the input. canon erases a global offset shared by both
    factors of a product, never a relative one."""
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

def linear_H_constraints(variables, span, t, U):
    """<[H, O_v]> = 0 in any energy eigenstate. In-span forms are usable
    equalities (orbit-pairing theorem predicts ZERO at degree-4 span; a
    nonzero count means the theorem failed -- verify one by hand before
    trusting). Out-of-span forms are dropped."""
    kept, out_of_span = [], 0
    seen = set()
    for v in sorted(variables):
        form = to_linear_form(ad_H_op(shift_op(op_of(v), OX, OY), t, U))
        if not form:
            continue
        if set(form) <= span:
            h = _normalize(form)
            if h not in seen:
                seen.add(h)
                kept.append(form)
        else:
            out_of_span += 1
    return kept, out_of_span

# ============================================================
# 9. G blocks
# ============================================================

def build_G_block(cands, span, t, U):
    """<O_i^dag [H, O_j]> over shifted candidates. Two-stage filter:
    (1) diagonal prefilter -- a candidate whose own <O^dag [H,O]> leaves the
        span can never survive (deleting OTHERS can't repair its diagonal),
        and the diagonal costs one entry instead of a row;
    (2) greedy removal on the survivors to reach an in-span principal
        submatrix.
    Then symmetrize, harvesting antisymmetric parts (= <[H, O_i^dag O_j]>,
    valid eigenstate equalities). Returns (kept_original_indices, G, harvested)."""
    for o in cands:
        for k in o:
            assert n_charge(k) == 0, "number-changing G candidate"

    scands = [shift_op(o, OX, OY) for o in cands]
    hops = [ad_H_op(o, t, U) for o in scands]
    dags = [dagger(o) for o in scands]

    # stage 1: diagonal prefilter
    pre = [i for i in range(len(scands))
           if set(to_linear_form(multiply(dags[i], hops[i]))) <= span]
    log(f"  G prefilter: {len(pre)}/{len(cands)} pass the diagonal test")
    if not pre:
        return [], [], []

    # stage 2: full form fill on survivors, then greedy
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
                   if (v := Fraction(G[i][j].get(k, 0) + G[j][i].get(k, 0), 2)) != 0}
            G[i][j] = dict(sym)
            G[j][i] = dict(sym)
            if anti:
                h = _normalize(anti)
                if h not in seen:
                    seen.add(h)
                    harvested.append(anti)
    return keep, G, harvested

# ============================================================
# 10. Ward identities
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
            h = _normalize(form)
            if h not in seen:
                seen.add(h)
                out.append(form)
    return out, dropped

# ============================================================
# 11. Solver (MOSEK low-level, dual conic form)
# ============================================================

def assemble_and_solve(blocks, obj, equalities, var_index, do_max=False,
                       stream_log=False):
    import mosek

    nv = len(var_index)
    dims = [len(m) for m in blocks]
    nE = len(equalities)

    f = [0.0] * nv
    fconst = 0.0
    for k, v in obj.items():
        if k == 0:
            fconst = float(v)
        else:
            f[var_index[k]] = float(v)

    blockdata = []
    for m in blocks:
        n = len(m)
        a0 = ([], [], [])
        pervar = {}
        for i in range(n):
            for j in range(i + 1):
                for k, val in m[i][j].items():
                    tgt = a0 if k == 0 else pervar.setdefault(
                        var_index[k], ([], [], []))
                    tgt[0].append(i); tgt[1].append(j); tgt[2].append(float(val))
        blockdata.append((n, a0, pervar))

    def run(fvec):
        with mosek.Env() as env, env.Task() as task:
            if stream_log:
                task.set_Stream(mosek.streamtype.log,
                                lambda s: sys.stdout.write(s))
            task.appendcons(nv)
            task.appendvars(nE)
            for e in range(nE):
                task.putvarbound(e, mosek.boundkey.fr, 0.0, 0.0)
            task.appendbarvars(dims)
            for e, (form, rhs) in enumerate(equalities):
                cst = 0.0
                for k, v in form.items():
                    if k == 0:
                        cst = float(v)
                    else:
                        task.putaij(var_index[k], e, float(v))
                task.putcj(e, float(rhs) - cst)
            for i in range(nv):
                task.putconbound(i, mosek.boundkey.fx, fvec[i], fvec[i])
            for b, (n, a0, pervar) in enumerate(blockdata):
                if a0[0]:
                    idx = task.appendsparsesymmat(n, a0[0], a0[1], a0[2])
                    task.putbarcj(b, [idx], [-1.0])
                for vi, (ri, ci, vv) in pervar.items():
                    idx = task.appendsparsesymmat(n, ri, ci, vv)
                    task.putbaraij(vi, b, [idx], [1.0])
            task.putobjsense(mosek.objsense.maximize)
            t1 = time.time()
            task.optimize()
            solsta = task.getsolsta(mosek.soltype.itr)
            pobj = task.getprimalobj(mosek.soltype.itr)
            log(f"  solve: {time.time()-t1:.1f}s, status {solsta}, "
                f"certificate {pobj:.6f}")
            if solsta != mosek.solsta.optimal:
                log("  WARNING: not OPTIMAL")
            return pobj

    lower = run(f) + fconst
    upper = None
    if do_max:
        upper = -run([-fi for fi in f]) + fconst
    return lower, upper

# ============================================================
# 12. Main pipeline
# ============================================================

def moment_matrix(basis):
    dags = [dagger(o) for o in basis]
    return [[to_linear_form(multiply(di, bj)) for bj in basis] for di in dags]

def sanity_check_G():
    """Doublon diagonal of G must reproduce the old Mathematica G1 structure:
    +4t on the hop pattern, +U-structure on the degree-4 pattern. The hop
    sign is unambiguous; a zero or a -4 here means coordinate or sign rot."""
    doub = shift_op(u_part(), OX, OY)
    entry = to_linear_form(multiply(dagger(doub), ad_H_op(doub, 1, 8)))
    log(f"G sanity (doublon diagonal): {show_op(entry)}")
    assert entry, "G doublon diagonal is zero: coordinate bug is back"
    return entry

def solve_bootstrap(t=1, U=8, nu=Fraction(7, 8), use_G=True, do_max=False):
    t0 = time.time()
    Mbases = build_M_bases()
    Mmats = [moment_matrix(b) for b in Mbases]
    log(f"M fill: {time.time()-t0:.1f}s, blocks {[len(m) for m in Mmats]}")

    neutral, charged = split_variables(Mmats)
    span = neutral | charged | {0}

    hcons, h_oos = linear_H_constraints(neutral, span, t, U)
    log(f"linear H-constraints: {len(hcons)} kept (theorem predicts 0), "
        f"{h_oos} out-of-span")
    if hcons:
        log("  e.g. " + show_op(hcons[0]) + " = 0   <-- verify by hand")

    Gmats, g_lin = [], []
    if use_G:
        sanity_check_G()
        t0 = time.time()
        cands = build_G_candidates()
        keep, G, g_lin = build_G_block(cands, span, t, U)
        if keep:
            Gmats = [G]
        log(f"G: kept {len(keep)}/{len(cands)} ops, "
            f"{len(g_lin)} harvested equalities, {time.time()-t0:.1f}s")
        if keep:
            log("  ops: " + ", ".join(show(min(cands[i])) for i in keep[:6])
                + (" ..." if len(keep) > 6 else ""))

    allmats = Mmats + Gmats
    neutral2, charged2 = split_variables(allmats)
    assert neutral2 <= neutral and charged2 <= charged, \
        "G blocks escaped M span despite filter"

    wards, dropped = ward_constraints(neutral)
    prune_charged(allmats)

    obj = objective(t, U)
    fill = filling_constraint(nu)
    var_index = {k: i for i, k in enumerate(sorted(neutral))}
    assert all(k in var_index for k in obj)
    assert all(k == 0 or k in var_index for k in fill[0])

    equalities = ([(fill[0], fill[1])]
                  + [(w, 0) for w in wards]
                  + [(h, 0) for h in hcons]
                  + [(g, 0) for g in g_lin])

    log(f"variables: {len(var_index)} neutral, {len(charged)} charged (pruned)")
    log(f"equalities: 1 filling + {len(wards)} ward + {len(hcons)} H-linear "
        f"+ {len(g_lin)} G-antisym")
    log(f"objective: {show_op(obj)}")

    lower, upper = assemble_and_solve(allmats, obj, equalities, var_index,
                                      do_max=do_max)
    return lower, upper

if __name__ == "__main__":
    log("=== M only ===")
    lo_M, up_M = solve_bootstrap(use_G=False, do_max=True)
    log("\n=== M + G ===")
    lo_MG, up_MG = solve_bootstrap(use_G=True, do_max=True)
    log(f"\nM only:  [{lo_M:.6f}, {up_M:.6f}]")
    log(f"M + G:   [{lo_MG:.6f}, {up_MG:.6f}]")