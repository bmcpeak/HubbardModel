"""
Infinite-volume Hubbard bootstrap.

  M blocks:  <O^dag O>      >= 0   one block per N-charge sector, all degrees
                                    <= level on the patch, spins mixed (Sz
                                    pruning block-diagonalizes internally).
                                    Both q and -q sectors kept: P/Q-type
                                    conditions differ by anticommutator
                                    inhomogeneities, NOT redundant.
  G blocks:  <O^dag [H,O]>  >= 0   number-conserving O only.
  Equalities: filling, SU(2) Ward identities, in-span <[H,O]> = 0,
              G-antisymmetric parts. ALL equalities are Sz-scrubbed before
              assembly: charged EVs are zero by selection, so deleting
              charged keys from an equality imposes the rule, same as
              pruning PSD entries. (Skipping this for g_lin was the
              KeyError-as-giant-integer bug.)

All bootstrap content lives in Section 6 (CONFIGURATION).

Coordinate discipline: canon anchors at (0,0); ad_H grows support by one
ring, so its inputs are pre-shifted by (OX, OY), and BOTH factors of any
product must share the offset -- canon erases global shifts, not relative
ones. ad_H asserts interiority.

Representation: monomial = packed int (D << NMODES) | C; d's ascending,
c's descending; dagger = sign-free (D,C)->(C,D). Operator = dict
{key: coeff}. Linear form = dict {canonical_key: coeff}, key 0 = constant.

Physics assumptions: reality <O^dag> = <O>; singlet ground state (wards,
Sz selection); G and <[H,O]> = 0 valid in the ground state.
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
# 6. CONFIGURATION
# ============================================================

P = 2
PATCH = [(i, j) for i in range(P) for j in range(P)]

# Interior anchor for ad_H (adds one ring). Invariant: (P-1) + OX + 1 < W.
OX, OY = 2, 2

def set_patch(p):
    global P, PATCH
    P = p
    PATCH = [(i, j) for i in range(P) for j in range(P)]
    assert (P - 1) + OX + 1 < W, "window too small for this patch"
    # canon / _mono_product / ad_H caches are keyed on patterns, not on P:
    # nothing to clear.

def _pool(patch):
    return [(i, j, s) for (i, j) in patch for s in (0, 1)]

def monomials(patch, nd, nc):
    pool = _pool(patch)
    return [op_of(pack_sites(D, C))
            for D in combinations(pool, nd)
            for C in combinations(pool, nc)]

def build_M_bases(level):
    assert level in (2, 3)
    q0  = [IDENTITY] + monomials(PATCH, 1, 1)
    qm1 = monomials(PATCH, 0, 1)
    qp1 = monomials(PATCH, 1, 0)
    qm2 = monomials(PATCH, 0, 2)
    qp2 = monomials(PATCH, 2, 0)
    bases = [q0, qm1, qp1, qm2, qp2]
    if level >= 3:
        qm1 += monomials(PATCH, 1, 2)
        qp1 += monomials(PATCH, 2, 1)
        bases += [monomials(PATCH, 0, 3), monomials(PATCH, 3, 0)]
    return bases

def build_G_candidates():
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

def scrub_forms(forms):
    """Delete Sz-charged keys from equality forms (their EVs are zero by
    selection, so deletion imposes the rule). Drop forms that empty out."""
    out = []
    for f in forms:
        f2 = {k: v for k, v in f.items() if k == 0 or sz_charge(k) == 0}
        if f2:
            out.append(f2)
    return out

# ============================================================
# 8. Hamiltonian
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

def show_mma(key):
    """Monomial in Mathematica NCM notation, 1-indexed sites."""
    if key == 0:
        return "1"
    Dsites, Csites = sites_of(key >> NMODES), sites_of(key & CMASK)
    parts  = [f"d[{i+1},{j+1},{s}]" for i, j, s in Dsites]
    parts += [f"c[{i+1},{j+1},{s}]" for i, j, s in reversed(Csites)]
    return "**".join(parts)

def report_H_constraints(variables, span, t, U, path="hcons_check.m"):
    """For each kept <[H,O_v]> = 0, write the generator v and the constraint
    form in Mathematica syntax. Each form entry is a CANONICAL pattern: to
    evaluate in an ED state, average the pattern's EV over all translations
    and the 32 group images (with signs), per the canon definition --
    equivalently, embed the pattern anywhere and symmetrize the state's EV.
    The generator O_v is reported UNSHIFTED (anchored at (1,1))."""
    lines = ["(* <[H,O]> = 0 constraints; evaluate each form == 0 in an",
             "   exact eigenstate. Site indices are 1-based. *)"]
    idx = 0
    seen = set()
    for v in sorted(variables):
        form = to_linear_form(ad_H_op(shift_op(op_of(v), OX, OY), t, U))
        if not form or not set(form) <= span:
            continue
        h = _normalize(form)
        if h in seen:
            continue
        seen.add(h)
        idx += 1
        lines.append(f"\n(* constraint {idx}: generator O = {show_mma(v)} *)")
        terms = " + ".join(
            f"({v2})*ev[{show_mma(k)}]" for k, v2 in sorted(form.items()))
        lines.append(f"hcons[{idx}] = {terms};")
    lines.append(f"\nnhcons = {idx};")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    log(f"wrote {idx} constraints to {path}")

# ============================================================
# 9. G blocks
# ============================================================

def build_G_block(cands, span, t, U):
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
                       min_too=True, stream_log=False):
    """blocks: matrices of linear forms (M and G alike). Poses the Lagrangian
    dual in the Optimizer API; the task primal at any feasible point
    certifies a bound. min_too=False skips the minimization (for max-only
    runs); returns (lower_or_None, upper_or_None)."""
    import mosek

    nv = len(var_index)
    dims = [len(m) for m in blocks]
    nE = len(equalities)

    for form, _ in equalities:
        for k in form:
            assert k == 0 or k in var_index, \
                f"equality references non-variable {show(k)} (Sz-scrub missing?)"

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
                    assert k == 0 or k in var_index, \
                        f"block entry references non-variable {show(k)}"
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
            task.putdouparam(mosek.dparam.optimizer_max_time, 5400.0)
            task.putobjsense(mosek.objsense.maximize)
            t1 = time.time()
            task.optimize()
            if stream_log:
                sys.stdout.flush()
            solsta = task.getsolsta(mosek.soltype.itr)
            pobj = task.getprimalobj(mosek.soltype.itr)
            log(f"  solve: {time.time()-t1:.1f}s, status {solsta}, "
                f"certificate {pobj:.6f}")
            if solsta != mosek.solsta.optimal:
                log("  WARNING: not OPTIMAL")
            return pobj

    lower = run(f) + fconst if min_too else None
    upper = None
    if do_max:
        upper = -run([-fi for fi in f]) + fconst
    return lower, upper

# ============================================================
# 12. Pipeline
# ============================================================

def moment_matrix(basis, label=None):
    n = len(basis)
    dags = [dagger(o) for o in basis]
    rows = []
    t0 = time.time()
    for i in range(n):
        rows.append([to_linear_form(multiply(dags[i], bj)) for bj in basis])
        if label and n >= 500 and (i + 1) % 100 == 0:
            log(f"    {label}: row {i+1}/{n} ({time.time()-t0:.0f}s)")
    return rows

def sanity_check_G():
    doub = shift_op(u_part(), OX, OY)
    entry = to_linear_form(multiply(dagger(doub), ad_H_op(doub, 1, 8)))
    assert entry, "G doublon diagonal is zero: coordinate bug is back"




def solve_bootstrap(level=3, t=1, U=8, nu=Fraction(7, 8), use_G=True,
                    do_max=False, min_too=True, stream_log=False):
    t0 = time.time()
    Mbases = build_M_bases(level)
    Mmats = [moment_matrix(b, label=f"block {i}")
             for i, b in enumerate(Mbases)]
    log(f"M fill (P={len(PATCH)**0 and 0 or int(len(PATCH)**.5)}, "
        f"level {level}): {time.time()-t0:.1f}s, "
        f"blocks {[len(m) for m in Mmats]}")

    neutral, charged = split_variables(Mmats)
    span = neutral | charged | {0}

    t0 = time.time()
    hcons, h_oos = linear_H_constraints(neutral, span, t, U)
    log(f"linear H-constraints: {len(hcons)} kept, {h_oos} out-of-span "
        f"({time.time()-t0:.1f}s)")

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

    allmats = Mmats + Gmats
    neutral2, charged2 = split_variables(allmats)
    assert neutral2 <= neutral and charged2 <= charged, \
        "G blocks escaped M span despite filter"

    wards, dropped = ward_constraints(neutral)
    log(f"wards: {len(wards)} kept, {dropped} dropped")

    prune_charged(allmats)
    hcons = scrub_forms(hcons)
    g_lin = scrub_forms(g_lin)
    wards = scrub_forms(wards)

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

    return assemble_and_solve(allmats, obj, equalities, var_index,
                              do_max=do_max, min_too=min_too,
                              stream_log=stream_log)

# ============================================================
# 13. Scan driver
# ============================================================

def run_scan():
    # Ordered so the config the scan exists for (P=3 level=3, G off -- first
    # configuration where the H-derived constraints can fire) lands before
    # the more fragile G variants. MOSEK's iteration log streams for the
    # P=3 level=3 solves only.
    configs = [
        (2, 2, False), (2, 2, True),
        (2, 3, False), (2, 3, True),
        (3, 2, False), (3, 2, True),
        (3, 3, False), (3, 3, True),
    ]
    results = []
    for p, level, use_G in configs:
        set_patch(p)
        big = (p == 3 and level == 3)
        tag = f"P={p} level={level} G={'on ' if use_G else 'off'}"
        log(f"\n=== {tag} ===")
        t0 = time.time()
        try:
            lo, up = solve_bootstrap(level=level, use_G=use_G, do_max=True,
                                     stream_log=big)
            results.append((tag, lo, up, time.time() - t0, "ok"))
        except Exception as e:
            results.append((tag, None, None, time.time() - t0,
                            f"FAILED: {type(e).__name__}: {e}"))
            log(f"  FAILED after {time.time()-t0:.0f}s: "
                f"{type(e).__name__}: {e}")

    log("\n" + "=" * 74)
    log(f"{'config':24s} {'lower':>12s} {'upper(max)':>12s} {'wall':>9s}")
    log("-" * 74)
    for tag, lo, up, wall, status in results:
        if lo is None:
            log(f"{tag:24s} {status}")
        else:
            log(f"{tag:24s} {lo:>12.6f} {up:>12.6f} {wall:>8.0f}s")
    log("-" * 74)
    log(f"{'AFQMC reference':24s} {-0.767:>12.3f}")

import pickle, os

if __name__ == "__main__":
    import json

    set_patch(3)
    results = {}

    def bank(key, val):
        results[key] = val
        with open("p3l3_bounds.json", "w") as fh:
            json.dump(results, fh, indent=1)
        log(f"BANKED {key} = {val:.6f}")

    # Priority order: each solve banks to disk on completion, so killing
    # the run at any point keeps everything finished so far.

    log("=== 1) max with G ===")
    _, up_G = solve_bootstrap(level=3, use_G=True, do_max=True,
                              min_too=False, stream_log=True)
    bank("upper_max_state_G", up_G)

    log("=== 2) max without G ===")
    _, up = solve_bootstrap(level=3, use_G=False, do_max=True,
                            min_too=False, stream_log=True)
    bank("upper_max_state", up)

    log("=== 3) min with G ===")
    lo_G, _ = solve_bootstrap(level=3, use_G=True, do_max=False,
                              stream_log=True)
    bank("lower_G", lo_G)

    log("=== 4) min without G (regression: -0.938438) ===")
    lo, _ = solve_bootstrap(level=3, use_G=False, do_max=False,
                            stream_log=True)
    bank("lower", lo)

    log("\n" + "=" * 50)
    for k, v in results.items():
        log(f"{k:22s} {v:.6f}")

#if __name__ == "__main__":
#    run_scan()