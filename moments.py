"""Moment variables: interning expectation values of canonical monomials.

The SDP's variables are expectation values <m> of monomials in the
translation-invariant, D4-invariant, spin-flip-invariant, real ground
state. This module reduces the enormous set of raw monomials produced by
operator products to the (much smaller) set of independent moment
variables, in two stages applied at intern time:

  1. SELECTION RULES (exact zeros, never become variables):
       - U(1) x U(1): charges(m) != (0, 0)  =>  <m> = 0;
       - symmetry kill: the orbit of m reaches its canonical representative
         with both signs  =>  <m> = -<m> = 0 (canonical_moment sign 0).
  2. CANONICALIZATION: surviving monomials map to (moment_id, sign) via
     canonical_moment (translations x D4 x flip x dagger).

Ward identities are a THIRD reduction — linear relations among surviving
moments from <[H, O]> = 0 — and live in wards.py; they are relations, not
identifications, and require elimination rather than interning.

MomentTable is the memo layer: canonical_moment is pure and uncached by
design; each distinct raw monomial is canonicalized exactly once here.
The raw memo is the dominant memory cost at scale (it grows with distinct
raw monomials, not with moment variables); drop_raw_memo() frees it once
assembly is done, keeping only the id <-> canonical-monomial tables.

The key downstream primitive is ``resolve(op) -> {moment_id: coeff}``:
the expectation functional of an Operator as a sparse row over moment
variables. A cached resolve row of a† L^k b IS the Hankel entry Y_k(a,b);
assemble.py never needs anything finer.
"""

from __future__ import annotations

from algebra import Operator, IDENTITY_MONO
from modes import Lattice
from symmetry import canonical_moment


class MomentTable:
    def __init__(self, lat: Lattice, use_flip: bool = True,
                 use_dagger: bool = True):
        self.lat = lat
        self.use_flip = use_flip
        self.use_dagger = use_dagger
        self.frozen = False
        self.monos: list = []      # id -> canonical monomial
        self._canon: dict = {}     # canonical monomial -> id
        self._raw: dict = {}       # raw monomial -> (id | None, sign) memo
        # the identity moment <1> = 1 always exists; give it id 0
        self.identity_id = self._intern_canonical(IDENTITY_MONO)

    # -- internals ---------------------------------------------------------

    def _intern_canonical(self, mc) -> int:
        mid = self._canon.get(mc)
        if mid is None:
            if self.frozen:
                raise RuntimeError(
                    "MomentTable is frozen; a resolve after problem "
                    "emission tried to intern a NEW moment — this would "
                    "silently desynchronize SDPProblem.nvars")
            mid = len(self.monos)
            self.monos.append(mc)
            self._canon[mc] = mid
        return mid

    def freeze(self):
        """Forbid interning new moments; lookups of known moments still work.
        Call after assemble_problem and before solving."""
        self.frozen = True

    # -- lookup -------------------------------------------------------------

    def lookup(self, m):
        """(moment_id, sign) with <m> = sign * moment[moment_id].

        (None, 0) if <m> = 0 exactly by a selection rule or symmetry kill.
        Memoized per raw monomial.
        """
        hit = self._raw.get(m)
        if hit is not None:
            return hit
        if self.lat.charges(m) != (0, 0):
            out = (None, 0)
        else:
            mc, sign = canonical_moment(self.lat, m, use_flip=self.use_flip,
                                        use_dagger=self.use_dagger)
            out = (None, 0) if sign == 0 else (self._intern_canonical(mc), sign)
        self._raw[m] = out
        return out

    def resolve(self, op: Operator) -> dict:
        """Expectation functional of op: sparse row {moment_id: coeff}.

        <op> = sum_id row[id] * moment[id]. Coefficient exactness is
        preserved (int/Fraction in, int/Fraction out).
        """
        row: dict = {}
        for m, v in op.terms.items():
            mid, sign = self.lookup(m)
            if mid is None:
                continue
            w = row.get(mid, 0) + (v if sign > 0 else -v)
            if w:
                row[mid] = w
            elif mid in row:
                del row[mid]
        return row

    # -- bookkeeping ---------------------------------------------------------

    def __len__(self):
        return len(self.monos)

    def drop_raw_memo(self):
        """Free the raw-monomial memo (the big table) after assembly."""
        n = len(self._raw)
        self._raw = {}
        return n

    def stats(self):
        return {"moments": len(self.monos), "raw_memo": len(self._raw)}
