"""
algebra.py -- fermionic operator algebra for the Hubbard bootstrap.
FROZEN LAYER: covered by test_oracle.py; do not edit without rerunning it.

Representation:
  - Monomial = packed int (D << NMODES) | C over window modes.
    d's in ascending mode order, c's in DESCENDING; under this convention
    dagger is the sign-free swap (D, C) -> (C, D).
  - Operator = dict {key: coeff}, normal-ordered monomials only.
  - Linear form = dict {canonical_key: coeff}; key 0 = the identity,
    always the constant term, never an SDP variable.

Canonicalization: orbit-minimum over Z^2 (by anchoring) semidirect
(D4 x spinflip x dagger), fermionic signs by inversion counting.
canon defines the VARIABLE set (state-invariance <gX> = <X>); it is
independent of, and required by, any irrep organization of operator
bases (Schur block-diagonalization of moment matrices holds as an
identity of forms only because canon has identified the orbits).

Coordinate discipline: canon anchors at (0,0). Any support-growing
operation (ad_H lives in physics.py) needs pre-shifted interior input,
and BOTH factors of any product must share the offset -- canon erases
global shifts, never relative ones.
"""

from functools import lru_cache
from fractions import Fraction

# ============================================================
# Window and modes
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
# Packing
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

def support_sites(key):
    D, C = unpack(key)
    return {(i, j) for i, j, s in sites_of(D | C)}

# ============================================================
# Kernels
# ============================================================

def mul_c(terms, m):
    """Right-multiply operator by c_m (annihilation)."""
    out = {}
    bit = 1 << m
    below = bit - 1
    for key, coeff in terms.items():
        D, C = key >> NMODES, key & CMASK
        if C & bit:
            continue                          # c_m^2 = 0
        sign = -1 if (C & below).bit_count() & 1 else 1
        k = (D << NMODES) | (C | bit)
        out[k] = out.get(k, 0) + sign * coeff
    return out

def mul_d(terms, m):
    """Right-multiply operator by d_m (creation)."""
    out = {}
    bit = 1 << m
    below = bit - 1
    for key, coeff in terms.items():
        D, C = key >> NMODES, key & CMASK
        if C & bit:                           # c_m d_m = 1 - d_m c_m
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
    """Elementary factor sequence: d's ascending, then c's descending."""
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
    """Hermitian conjugate: sign-free (D,C) -> (C,D) under our convention.
    NOTE for symmetry.py: dagger maps charge-q bases to charge-(-q) bases;
    it acts WITHIN canon (variable identification), not within irrep blocks."""
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
# Symmetry group elements and canonicalization
# ============================================================
# PG8, FINITE, act, parity_sign are shared with symmetry.py (which builds
# representation matrices from the same primitives) -- they live here so
# the two layers cannot drift.

PG8 = [
    (( 1, 0), ( 0, 1)), (( 0,-1), ( 1, 0)), ((-1, 0), ( 0,-1)), (( 0, 1), (-1, 0)),
    (( 1, 0), ( 0,-1)), ((-1, 0), ( 0, 1)), (( 0, 1), ( 1, 0)), (( 0,-1), (-1, 0)),
]
PG_NAMES = ["e", "r90", "r180", "r270", "mj", "mi", "diag", "adiag"]

FINITE = [(M, sf, dag) for M in PG8 for sf in (0, 1) for dag in (0, 1)]  # 32

def act(M, sf, p):
    (a, b), (c_, d_) = M
    i, j, s = p
    return (a * i + b * j, c_ * i + d_ * j, s ^ sf)

def parity_sign(sites):
    """Parity of the permutation sorting the mode-index sequence. Only
    relative signs between original and image matter; the fixed reversal
    between c-storage order (descending) and traversal order (ascending)
    cancels between original and image."""
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
    """-> (canonical_key, sign). sign = 0: EV forced to zero (some group
    element maps the canonical rep to itself with -1)."""
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

def normalize_form(form):
    """Hashable scale-invariant representation, for deduplication."""
    kmin = min(form)
    lead = Fraction(form[kmin])
    return tuple(sorted((k, Fraction(v) / lead) for k, v in form.items()))

# ============================================================
# Charges
# ============================================================

def n_charge(key):
    D, C = unpack(key)
    return D.bit_count() - C.bit_count()

def sz_charge(key):
    D, C = unpack(key)
    q = sum(1 if s == UP else -1 for _, _, s in sites_of(D))
    q -= sum(1 if s == UP else -1 for _, _, s in sites_of(C))
    return q

# ============================================================
# Printing
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