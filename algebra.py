"""Canonical fermionic monomial algebra over abstract modes.

Representation
--------------
A monomial is a pair of arbitrary-precision Python ints ``(dag, c)``:
bit ``i`` of ``dag`` set  <=>  a†_i present; bit ``i`` of ``c``  <=>  a_i present.
The pair encodes the *canonical string*

    a†_{i1} ... a†_{ip}  a_{jq} ... a_{j1}

with daggers ascending left-to-right (i1 < ... < ip) and annihilators
DESCENDING left-to-right (j1 < ... < jq, smallest rightmost).

Consequences of this convention:
  * adjoint:  (dag, c)† = (c, dag)  with sign exactly +1;
  * a monomial key is unique (membership + fixed order = canonical form);
  * sign bookkeeping reduces to popcount parities on mask regions.

``dag`` and ``c`` may overlap (e.g. the number operator n_i = a†_i a_i is
``(1<<i, 1<<i)``); repeated modes *within* one mask are impossible by
construction, which is Pauli exclusion at the representation level.

Mode indices are abstract nonnegative ints. The lattice embedding
(mode = 2*(W*y + x) + spin, translations, D4, ...) lives in modes.py /
symmetry.py and is deliberately invisible here: this file is pure CAR algebra.

Coefficients are duck-typed: use int / fractions.Fraction for exact work
(Ward elimination), float once you hand things to the SDP layer. All
construction paths from a Hubbard H with rational t, U stay exact.

Memory note: the dict-of-terms form here is the *build* form. A frozen
structure-of-arrays form (sorted packed keys + coefficient array) will be
added when profiling says so; every public function below takes and returns
plain ``(dag, c)`` keys and dicts precisely so that swap stays local.

Requires Python >= 3.10 (int.bit_count).
"""

from __future__ import annotations


# ----------------------------------------------------------------------
# bit utilities
# ----------------------------------------------------------------------

def parity(x: int) -> int:
    """Parity (0/1) of the popcount of x."""
    return x.bit_count() & 1


def bits_ascending(x: int):
    """Yield positions of set bits of x, smallest first."""
    while x:
        lsb = x & -x
        yield lsb.bit_length() - 1
        x ^= lsb


def bits_descending(x: int):
    """Yield positions of set bits of x, largest first."""
    while x:
        i = x.bit_length() - 1
        yield i
        x ^= 1 << i


# ----------------------------------------------------------------------
# elementary right-multiplications (the sign kernel)
# ----------------------------------------------------------------------
# Everything else is built from these two functions; they carry ALL sign
# conventions. Verified against a dense Jordan-Wigner representation in
# tests/test_algebra.py.

def mul_c(dag: int, c: int, i: int):
    """(dag, c) * a_i  ->  (sign, (dag, c'))  or  None if identically zero.

    a_i is appended at the far right and moved left to its slot in the
    descending annihilator string: it passes exactly the annihilators with
    mode < i, each costing a factor (-1).
    """
    bit = 1 << i
    if c & bit:
        return None  # a_i a_i = 0
    sign = -1 if parity(c & (bit - 1)) else 1
    return sign, (dag, c | bit)


def mul_d(dag: int, c: int, i: int):
    """(dag, c) * a†_i  ->  list of (sign, (dag', c')) with <= 2 entries.

    a†_i is appended at the far right and moved left through the whole
    annihilator string. If a_i is present, meeting it produces the
    contraction term (a_i a†_i = 1 - a†_i a_i): a†_i passes the
    #{x in c : x < i} smaller annihilators sitting to the right of a_i,
    then the pair cancels. The pass-through term always exists unless
    a†_i is already in dag: a†_i passes ALL of c (sign (-1)^|c|), enters
    the dagger string from the right and passes the daggers with mode > i.
    """
    out = []
    bit = 1 << i
    if c & bit:
        s = -1 if parity(c & (bit - 1)) else 1
        out.append((s, (dag, c & ~bit)))
    if not (dag & bit):
        s = -1 if (parity(c) ^ parity(dag >> (i + 1))) else 1
        out.append((s, (dag | bit, c)))
    return out


# ----------------------------------------------------------------------
# monomial-level operations
# ----------------------------------------------------------------------

def mono_dag(m):
    """Adjoint of a canonical monomial: (dag, c) -> (c, dag), sign +1."""
    return (m[1], m[0])


def mono_degree(m) -> int:
    return m[0].bit_count() + m[1].bit_count()


def mono_support(m) -> int:
    """Mask of all modes touched by the monomial."""
    return m[0] | m[1]


def mono_charge(m) -> int:
    """Net particle number: #creations - #annihilations.

    Total U(1) charge is intrinsic to the CAR algebra; per-spin charges
    live in modes.py where the spin embedding is defined.
    """
    return m[0].bit_count() - m[1].bit_count()


def mono_parity(m) -> int:
    """Fermion parity (0 = even, 1 = odd) of the monomial.

    Even operators on disjoint support commute exactly — the fact the
    locality filter in hamiltonian.py rests on.
    """
    return mono_degree(m) & 1


def mono_mul(m1, m2) -> dict:
    """Product of two canonical monomials.

    Returns {monomial: integer_coefficient} — the normal-ordered expansion.
    Implemented by peeling m2's string left-to-right (daggers ascending,
    then annihilators descending) and right-multiplying one elementary
    operator at a time; contractions branch, so term count is
    O(2^|overlap|) with overlap = c(m1) & dag(m2). Correctness lives
    entirely in mul_c / mul_d.
    """
    d2, c2 = m2
    terms = {m1: 1}
    for i in bits_ascending(d2):
        new = {}
        for (dg, cc), s in terms.items():
            for s2, key in mul_d(dg, cc, i):
                v = new.get(key, 0) + s * s2
                if v:
                    new[key] = v
                elif key in new:
                    del new[key]
        if not new:
            return {}
        terms = new
    for i in bits_descending(c2):
        new = {}
        for (dg, cc), s in terms.items():
            r = mul_c(dg, cc, i)
            if r is None:
                continue
            s2, key = r
            v = new.get(key, 0) + s * s2
            if v:
                new[key] = v
            elif key in new:
                del new[key]
        if not new:
            return {}
        terms = new
    return terms


def mono_str(m) -> str:
    """Human-readable form, e.g. 'd1 d3 c2 c0'."""
    dag, c = m
    parts = [f"d{i}" for i in bits_ascending(dag)]
    parts += [f"c{i}" for i in bits_descending(c)]
    return " ".join(parts) if parts else "1"


IDENTITY_MONO = (0, 0)


# Cached product path. Measured hit rates (2026-07): ~13% in the Krylov/L
# workload (useless -- L stays on the uncached mono_mul above), ~85% in
# moment-matrix assembly over symmetrized bases (projection spreads shared
# monomials over 16-element orbits). Hence: bounded cache, used by
# Operator.__mul__ only. Entries are immutable tuples; maxsize bounds worst-
# case memory to a few hundred MB.

from functools import lru_cache


@lru_cache(maxsize=1 << 19)
def mono_mul_items(m1, m2):
    """Cached, immutable view of mono_mul(m1, m2).items()."""
    return tuple(mono_mul(m1, m2).items())


def clear_product_cache():
    mono_mul_items.cache_clear()


# ----------------------------------------------------------------------
# Operator: linear combination of canonical monomials
# ----------------------------------------------------------------------

class Operator:
    """Sparse linear combination of canonical monomials.

    terms: dict {(dag, c): coeff}. Zero coefficients are never stored.
    Scalar type is whatever you feed in (int / Fraction / float / complex).
    """

    __slots__ = ("terms",)

    def __init__(self, terms=None):
        self.terms = {}
        if terms:
            for k, v in terms.items():
                if v:
                    self.terms[k] = v

    # -- constructors -------------------------------------------------

    @staticmethod
    def zero():
        return Operator()

    @staticmethod
    def identity(coeff=1):
        return Operator({IDENTITY_MONO: coeff})

    @staticmethod
    def monomial(m, coeff=1):
        return Operator({m: coeff})

    # -- ring structure ------------------------------------------------

    def __add__(self, other):
        out = dict(self.terms)
        for k, v in other.terms.items():
            w = out.get(k, 0) + v
            if w:
                out[k] = w
            elif k in out:
                del out[k]
        return Operator(out)

    def __sub__(self, other):
        out = dict(self.terms)
        for k, v in other.terms.items():
            w = out.get(k, 0) - v
            if w:
                out[k] = w
            elif k in out:
                del out[k]
        return Operator(out)

    def __neg__(self):
        return Operator({k: -v for k, v in self.terms.items()})

    def scale(self, s):
        if not s:
            return Operator()
        return Operator({k: s * v for k, v in self.terms.items()})

    def __rmul__(self, s):
        return self.scale(s)

    def __radd__(self, other):
        # so that sum(list_of_operators) works (sum starts from 0)
        if other == 0:
            return self
        return NotImplemented

    def __truediv__(self, s):
        return self.scale(1 / s)

    def __mul__(self, other):
        if not isinstance(other, Operator):
            return self.scale(other)
        acc: dict = {}
        for m1, v1 in self.terms.items():
            for m2, v2 in other.terms.items():
                v12 = v1 * v2
                for key, s in mono_mul_items(m1, m2):
                    w = acc.get(key, 0) + v12 * s
                    if w:
                        acc[key] = w
                    elif key in acc:
                        del acc[key]
        return Operator(acc)

    def dag(self):
        # every sane coefficient type (int, float, complex, Fraction,
        # numpy scalars, sympy) implements .conjugate(); no guard needed
        return Operator({(c, d): v.conjugate()
                         for (d, c), v in self.terms.items()})

    # -- convenience ----------------------------------------------------

    def __len__(self):
        return len(self.terms)

    def __bool__(self):
        return bool(self.terms)

    def __eq__(self, other):
        return isinstance(other, Operator) and self.terms == other.terms

    def chop(self, eps=1e-14):
        """Drop inexact (float/complex) coefficients with |v| < eps.

        No-op for exact types (int, Fraction): exact zeros are already
        never stored, and a small exact coefficient is information.
        """
        return Operator({k: v for k, v in self.terms.items()
                         if not isinstance(v, (float, complex)) or abs(v) >= eps})

    def max_degree(self):
        return max((mono_degree(m) for m in self.terms), default=0)

    def support(self):
        s = 0
        for m in self.terms:
            s |= mono_support(m)
        return s

    def is_hermitian(self):
        return self == self.dag()

    def __repr__(self):
        if not self.terms:
            return "0"
        bits = []
        for m, v in sorted(self.terms.items()):
            bits.append(f"({v})*{mono_str(m)}")
        return " + ".join(bits)


def commutator(a: Operator, b: Operator) -> Operator:
    return a * b - b * a


def anticommutator(a: Operator, b: Operator) -> Operator:
    return a * b + b * a


# single-mode generators, as Operators

def dop(i: int, coeff=1) -> Operator:
    """Creation operator a†_i."""
    return Operator({(1 << i, 0): coeff})


def cop(i: int, coeff=1) -> Operator:
    """Annihilation operator a_i."""
    return Operator({(0, 1 << i): coeff})


def nop(i: int, coeff=1) -> Operator:
    """Number operator n_i = a†_i a_i."""
    return Operator({(1 << i, 1 << i): coeff})
