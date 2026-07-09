"""
Fermionic operator algebra for the infinite-volume Hubbard bootstrap.

Representation:
  - A monomial is a packed integer: (D << NMODES) | C, where D, C are bitmasks
    over modes. Convention: d's in ascending mode order, c's in descending.
    Under this convention dagger is the sign-free swap (D, C) -> (C, D).
  - An operator is a dict {key: coeff}. Only normal-ordered monomials are stored.
  - An SDP "linear form" is a dict {canonical_key: coeff}; key 0 (the identity)
    is the constant term, never a variable.

Physics assumptions (flag these in any writeup):
  - Reality <O^dag> = <O> (real H, time reversal): dagger is in the symmetry group.
  - Singlet ground state (S^pm |psi> = 0): validity of the Ward identities.
  - Sz selection rule: Sz-charged EVs vanish; imposed by pruning PSD entries.

Sections:
  1. Window and mode indexing
  2. Packing / unpacking
  3. Algebra kernels
  4. Symmetry group and canonicalization
  5. Pretty printing
  6. Bootstrap bases and moment matrices
  7. Charges, pruning, variable splitting
  8. Objective, filling, Ward identities
  9. Mathematica export
 10. SDP assembly (MOSEK)
 11. Main
"""

import sys
import time
from itertools import combinations
from functools import lru_cache
from fractions import Fraction

def log(msg):
    print(msg, flush=True)      # PyCharm buffers; flush or misdiagnose hangs

# ============================================================
# 1. Window and mode indexing
# ============================================================

W = 10
NMODES = 2 * W * W
CMASK = (1 << NMODES) - 1

UP, DN = 1, 0   # S^+ = d[.,UP] c[.,DN]

def mode(i, j, s):
    assert 0 <= i < W and 0 <= j < W, f"site ({i},{j}) outside window"
    return 2 * (W * i + j) + s

def site_of_mode(m):
    s = m & 1
    m >>= 1
    return (m // W, m % W, s)

# ============================================================
# 2. Packing / unpacking
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

# ============================================================
# 3. Algebra kernels
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
    """Product of two coefficient-1 monomials, as a tuple of (key, coeff).
    Cached: moment-matrix fills hit the same pairs constantly."""
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
    (( 1, 0), ( 0, 1)),
    (( 0,-1), ( 1, 0)),
    ((-1, 0), ( 0,-1)),
    (( 0, 1), (-1, 0)),
    (( 1, 0), ( 0,-1)),
    ((-1, 0), ( 0, 1)),
    (( 0, 1), ( 1, 0)),
    (( 0,-1), (-1, 0)),
]
PG_NAMES = ["e", "r90", "r180", "r270", "mj", "mi", "diag", "adiag"]

FINITE = [(M, sf, dag) for M in PG8 for sf in (0, 1) for dag in (0, 1)]  # 32

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
        sign = parity_sign(Ds) * parity_sign(Cs)
        out.append((pack_sites(Ds, Cs), sign))
    return out

@lru_cache(maxsize=None)
def canon(key):
    """-> (canonical_key, sign). sign = 0 means the EV is forced to zero."""
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

# ============================================================
# 5. Pretty printing
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

def canon_verbose(key):
    print(f"input: {show(key)}")
    images = _images(key)
    for (M, sf, dag), (k2, sign) in zip(FINITE, images):
        label = PG_NAMES[PG8.index(M)] + (" sf" if sf else "") + (" dag" if dag else "")
        print(f"  {label:14s} {'+' if sign > 0 else '-'} {show(k2)}")
    kmin = min(k for k, _ in images)
    mult = sum(1 for k, _ in images if k == kmin)
    assert len(FINITE) % mult == 0, "orbit-stabilizer violated"
    signs = {s for k, s in images if k == kmin}
    result = (kmin, 0) if len(signs) == 2 else (kmin, signs.pop())
    tag = "  (forced zero)" if result[1] == 0 else ""
    print(f"canonical: {'+' if result[1] > 0 else '-' if result[1] < 0 else '0'} "
          f"{show(kmin)}{tag}")
    return result

# ============================================================
# 6. Bootstrap bases and moment matrices (12 blocks, 3x3 patch)
# ============================================================
# b11 carries no identity: after Sz pruning its identity row is structurally
# zero. b12's identity is where <1> and the density EVs enter the PSD structure.

PATCH = [(i, j) for i in range(3) for j in range(3)]

def build_bases():
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

    bases = [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12]
    assert [len(b) for b in bases] == [9, 9, 9, 9, 36, 81, 36, 36, 81, 36, 162, 163]
    return bases

def moment_matrix(basis):
    n = len(basis)
    dags = [dagger(o) for o in basis]
    return [[to_linear_form(multiply(dags[i], basis[j]))
             for j in range(n)] for i in range(n)]

def all_variables(mats):
    vs = set()
    for m in mats:
        for row in m:
            for entry in row:
                vs.update(entry.keys())
    vs.discard(0)
    return vs

# ============================================================
# 7. Charges, pruning, variable splitting
# ============================================================

def n_charge(key):
    D, C = unpack(key)
    return D.bit_count() - C.bit_count()

def sz_charge(key):
    D, C = unpack(key)
    q = sum(1 if s == UP else -1 for _, _, s in sites_of(D))
    q -= sum(1 if s == UP else -1 for _, _, s in sites_of(C))
    return q

def split_variables(mats):
    vs = all_variables(mats)
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
# 8. Objective, filling, Ward identities
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

def local_spin_charge(sites, pm):
    S = {}
    for i, j in sites:
        if pm > 0:
            S = add(S, multiply(d_op(i, j, UP), c_op(i, j, DN)))
        else:
            S = add(S, multiply(d_op(i, j, DN), c_op(i, j, UP)))
    return S

def _normalize(form):
    kmin = min(form)
    lead = Fraction(form[kmin])
    return tuple(sorted((k, Fraction(v) / lead) for k, v in form.items()))

def ward_constraints(neutral_vars):
    """For each neutral variable v, [S^+_local, O_v] has Sz-charge +2; for each
    monomial mu in it, <[S^-_local, mu]> = 0 is a neutral linear identity,
    valid in a singlet state. Forms reaching outside the SDP variable span
    are dropped (keeping them would introduce a free variable, i.e. vacuous)."""
    span = set(neutral_vars) | {0}
    seen, out = set(), []
    dropped = 0
    for v in sorted(neutral_vars):
        raised = commutator(local_spin_charge(support_sites(v), +1), op_of(v))
        for mu in sorted(raised):
            assert sz_charge(mu) == 2, "charge bookkeeping broken"
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
# 8b. Hamiltonian adjoint action and commutator blocks
# ============================================================
# H = sum_r T_r h_r with local terms; for O with site support A, only terms
# touching A contribute to [H, O] (all H-terms are even => disjoint support
# commutes exactly). Returns (t_part, U_part) separately.

def _hop_terms_touching(sites):
    """All directed hop terms d_p c_q on bonds with an endpoint in `sites`.
    Each undirected bond contributes both directions."""
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
    """[H, O] for monomial O, as (t_part, U_part) tuples of dict items.
    t_part multiplies -t (sign of -t NOT included here); U_part multiplies U."""
    O = op_of(key)
    sites = support_sites(key)
    t_part = commutator(_hop_terms_touching(sites), O)
    u_part = commutator(_u_terms_touching(sites), O)
    return tuple(t_part.items()), tuple(u_part.items())

def ad_H_op(op, t, U):
    """[H, A] for an operator dict, couplings applied. -t convention here."""
    out = {}
    for key, v in op.items():
        tp, up = ad_H(key)
        for k, w in tp:
            out[k] = out.get(k, 0) - t * v * w
        for k, w in up:
            out[k] = out.get(k, 0) + U * v * w
    return {k: v for k, v in out.items() if v != 0}

def commutator_matrix(basis, t, U):
    """M_ij = <O_i^dag [H, O_j]> as linear forms. Valid PSD constraint for
    number-conserving bases only (else needs E0(N+-1) >= E0(N))."""
    for o in basis:
        for k in o:
            assert n_charge(k) == 0, \
                "commutator block on number-changing operator: invalid at fixed filling"
    n = len(basis)
    dags = [dagger(o) for o in basis]
    hops = [ad_H_op(o, t, U) for o in basis]
    return [[to_linear_form(multiply(dags[i], hops[j]))
             for j in range(n)] for i in range(n)]

def check_linear_H_constraints(variables, t, U):
    """<[H, O_v]> must vanish identically after canon for every degree-4
    variable at this level (the orbit-pairing theorem). Nonzero output here
    means a bug in ad_H or canon, not physics."""
    bad = []
    for v in sorted(variables):
        form = to_linear_form(ad_H_op(op_of(v), t, U))
        if form:
            bad.append((v, form))
    return bad

# ============================================================
# 9. Mathematica export (couplings symbolic; set them in Mathematica)
# ============================================================

def _mma_coeff(v):
    if isinstance(v, Fraction):
        return f"({v.numerator}/{v.denominator})" if v.denominator != 1 else str(v.numerator)
    return str(v)

def _mma_form(form):
    if not form:
        return "0"
    terms = []
    for k, v in sorted(form.items()):
        c = _mma_coeff(v)
        terms.append(c if k == 0 else f"{c}*f[{k}]")
    return " + ".join(terms)

def export_mathematica(path, mats, t_form, u_form, fill, wards, var_index):
    with open(path, "w") as fh:
        fh.write("(* generated by algebra.py -- infinite-volume Hubbard bootstrap *)\n\n")
        fh.write("vars = {" + ", ".join(f"f[{k}]" for k in sorted(var_index)) + "};\n\n")
        fh.write("objT = " + _mma_form(t_form) + ";\n")
        fh.write("objU = " + _mma_form(u_form) + ";\n")
        fh.write("obj[t_, u_] := t objT + u objU;\n\n")
        fh.write("fillingLHS = " + _mma_form(fill[0]) + ";\n")
        fh.write(f"fillingRHS = {_mma_coeff(fill[1])};\n\n")
        fh.write("wards = {\n  " +
                 ",\n  ".join(_mma_form(w) for w in wards) + "\n};\n\n")
        fh.write("mats = {\n")
        for bi, m in enumerate(mats):
            n = len(m)
            fh.write("  {\n")
            for i in range(n):
                row = ", ".join(_mma_form(m[i][j]) for j in range(n))
                fh.write("    {" + row + "}" + (",\n" if i < n - 1 else "\n"))
            fh.write("  }" + (",\n" if bi < len(mats) - 1 else "\n"))
        fh.write("};\n")
    log(f"wrote {path}")

# ============================================================
# 10. SDP assembly (MOSEK)
# ============================================================

def assemble_and_solve(mats, obj, fill, wards, var_index, do_max=False,
                       stream_log=False):
    """Poses the Lagrangian DUAL of the moment-matrix SDP in MOSEK's
    low-level Optimizer API:

        max  b.lam - sum_b <A0_b, Y_b>
        s.t. sum_b <A_ib, Y_b> + (B^T lam)_i = f_i   (one row per variable)
             Y_b >= 0 (PSD), lam free

    The task's primal objective at any feasible point is a certified lower
    bound on the physics energy density; at OPTIMAL it equals the moment-SDP
    minimum. Returns (lower, upper_or_None).
    """
    import mosek

    nv = len(var_index)
    dims = [len(m) for m in mats]
    eqs = [(fill[0], fill[1])] + [(w, 0) for w in wards]
    nE = len(eqs)

    # objective vector over x (constant term shifts the bound at the end)
    f = [0.0] * nv
    fconst = 0.0
    for k, v in obj.items():
        if k == 0:
            fconst = float(v)
        else:
            f[var_index[k]] = float(v)

    # per-block: A0 triplets and per-variable coefficient-matrix triplets,
    # lower triangle only (MOSEK symmat convention: subi >= subj)
    t0 = time.time()
    blockdata = []
    for m in mats:
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
    log(f"  triplet prep: {time.time()-t0:.1f}s")

    def run(fvec):
        with mosek.Env() as env, env.Task() as task:
            if stream_log:
                task.set_Stream(mosek.streamtype.log,
                                lambda s: sys.stdout.write(s))
            task.appendcons(nv)
            task.appendvars(nE)
            for e in range(nE):
                task.putvarbound(e, mosek.boundkey.fr, 0.0, 0.0)  # free
            task.appendbarvars(dims)

            # linear part B^T lam and objective coefficients b.lam
            for e, (form, rhs) in enumerate(eqs):
                cst = 0.0
                for k, v in form.items():
                    if k == 0:
                        cst = float(v)
                    else:
                        task.putaij(var_index[k], e, float(v))
                task.putcj(e, float(rhs) - cst)

            # constraint rows fixed to f_i
            for i in range(nv):
                task.putconbound(i, mosek.boundkey.fx, fvec[i], fvec[i])

            # bar terms
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
            dobj = task.getdualobj(mosek.soltype.itr)
            log(f"  solve: {time.time()-t1:.1f}s, status {solsta}, "
                f"certificate {pobj:.6f}, other side {dobj:.6f}")
            if solsta != mosek.solsta.optimal:
                log("  WARNING: not OPTIMAL; certificate value is still a "
                    "valid bound if primal-feasible, but check status")
            return pobj

    lower = run(f) + fconst

    upper = None
    if do_max:
        upper = -(run([-fi for fi in f])) + fconst

    return lower, upper

# ============================================================
# 11. Main
# ============================================================

def solve_bootstrap(t=1, U=8, nu=Fraction(7, 8), export=None):
    t0 = time.time()
    bases = build_bases()
    mats = [moment_matrix(b) for b in bases]
    log(f"matrix fill: {time.time()-t0:.1f}s "
        f"(canon cache: {canon.cache_info().currsize}, "
        f"product cache: {_mono_product.cache_info().currsize})")

    t0 = time.time()
    neutral, charged = split_variables(mats)
    wards, dropped = ward_constraints(neutral)
    prune_charged(mats)
    log(f"symmetry/ward: {time.time()-t0:.1f}s")

    t_form = to_linear_form(hop_part())
    u_form = to_linear_form(u_part())
    obj = objective(t, U)
    fill = filling_constraint(nu)
    var_index = {k: i for i, k in enumerate(sorted(neutral))}

    assert all(k in var_index for k in obj), "objective variable not in any block"
    assert all(k == 0 or k in var_index for k in fill[0]), \
        "filling variable not in any block"

    log(f"blocks: {[len(m) for m in mats]}")
    log(f"variables: {len(var_index)} neutral, {len(charged)} charged (pruned)")
    log(f"ward constraints: {len(wards)} kept, {dropped} dropped (outside span)")
    log(f"objective: {show_op(obj)}")
    log(f"filling: {show_op(fill[0])} = {fill[1]}")

    if export:
        export_mathematica(export, mats, t_form, u_form, fill, wards, var_index)


    lower, upper = assemble_and_solve(mats, obj, fill, wards, var_index,do_max=False)
    log(f"\ncertified lower bound on e_0:      {lower:.6f}")
    if upper is not None:
        log(f"max-energy-state density (sanity): {upper:.6f}")
    return lower, upper

if __name__ == "__main__":
    bases = build_bases()
    m = moment_matrix(bases[10])
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            assert m[i][j] == m[j][i], f"b11 entry ({i},{j}) != ({j},{i})"
    log("b11 hermiticity: ok")

    solve_bootstrap(export="hubbard_sdp.m")