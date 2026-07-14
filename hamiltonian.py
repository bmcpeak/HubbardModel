"""Hubbard Hamiltonian on the infinite square lattice, via its local terms.

H = -t sum_{<xy>, s} (a†_{x s} a_{y s} + h.c.) + U sum_x n_{x up} n_{x dn}

H is never materialized. It is represented by its local terms (one on-site
interaction per site, two forward bonds per site), and the Liouvillian

    L(O) = [H, O]

is computed lazily: only terms whose site support touches the support of O
contribute, because every local term is fermion-even and therefore commutes
with anything on disjoint support. The locality filter is exact, not an
approximation (verified against the full-window commutator in tests).

Chemical potential is deliberately absent: [N, O] is proportional to O
within each charge grade, so a mu-term adds no new Krylov directions;
filling enters the SDP as the constraint <n> = nu.

Boundary policy: L raises if the support of O touches the window edge,
because the thermodynamic-limit commutator would involve sites outside the
window. This is a hard error by design — silent truncation would corrupt
bounds undetectably. Place seeds centrally and size W for your Krylov depth.

Sign conventions inherit entirely from algebra.py; every term below is
built out of dop/cop/nop products, so no new sign bookkeeping is introduced.
"""

from __future__ import annotations

from fractions import Fraction

from algebra import Operator, commutator, dop, cop, nop
from modes import Lattice


class Hubbard:
    def __init__(self, lat: Lattice, t=1, U=8):
        """t, U may be int / Fraction (exact pipeline) or float."""
        self.lat = lat
        self.t = t
        self.U = U
        self._term_cache: dict = {}   # ('int'|'bond', x, y[, dir]) -> Operator

    # -- local terms ---------------------------------------------------------

    def interaction(self, x: int, y: int) -> Operator:
        """U n_up n_dn at site (x, y)."""
        lat = self.lat
        return self.U * (nop(lat.mode(x, y, 0)) * nop(lat.mode(x, y, 1)))

    def bond(self, x: int, y: int, direction: str) -> Operator:
        """-t sum_s (a†_{r s} a_{r+e s} + h.c.) for e = 'x' or 'y'.

        Raises if the far endpoint leaves the window.
        """
        lat = self.lat
        if direction == "x":
            nx, ny = x + 1, y
        elif direction == "y":
            nx, ny = x, y + 1
        else:
            raise ValueError(f"invalid bond direction: {direction!r}")
        if not lat.in_window(nx, ny):
            raise ValueError(f"bond ({x},{y})->{direction} leaves window W={lat.W}")
        out = Operator.zero()
        for s in (0, 1):
            hop = dop(lat.mode(x, y, s)) * cop(lat.mode(nx, ny, s))
            out = out + (-self.t) * (hop + hop.dag())
        return out

    def energy_density(self, x0: int, y0: int) -> Operator:
        """h anchored at (x0, y0): interaction + the two forward bonds.

        sum over all sites of translates of this operator is H (each bond
        counted once), so <H>/V = <h> for a translation-invariant state.
        This is the SDP objective.
        """
        return (self.interaction(x0, y0)
                + self.bond(x0, y0, "x") + self.bond(x0, y0, "y"))

    # -- locality-filtered Liouvillian ----------------------------------------

    def _term_keys_touching(self, sites):
        """Unique local-term keys whose support intersects the given site set.

        Keys: ('int', x, y) or ('bond', x, y, 'x'|'y') with (x, y) the bond's
        base site. A site (x, y) is touched by its own interaction, its two
        forward bonds, and the two backward bonds based at (x-1, y), (x, y-1).
        """
        keys = set()
        for (x, y) in sites:
            keys.add(("int", x, y))
            keys.add(("bond", x, y, "x"))
            keys.add(("bond", x, y, "y"))
            keys.add(("bond", x - 1, y, "x"))
            keys.add(("bond", x, y - 1, "y"))
        return keys

    def _term(self, key) -> Operator:
        if key[0] == "int":
            return self.interaction(key[1], key[2])
        return self.bond(key[1], key[2], key[3])

    def _check_interior(self, supp) -> None:
        """Support must sit in [1, W-2]^2 so all touching bonds exist."""
        lat = self.lat
        minx, miny, maxx, maxy = lat.bbox(supp)
        if minx < 1 or miny < 1 or maxx > lat.W - 2 or maxy > lat.W - 2:
            raise ValueError(
                f"support bbox ({minx},{miny},{maxx},{maxy}) touches window "
                f"edge (W={lat.W}); increase W or re-place seeds — refusing "
                f"to silently truncate [H, O]")

    def neighborhood_H(self, op: Operator) -> Operator:
        """Sum of all local terms touching the support of op.

        [neighborhood_H(op), op] == [H, op] exactly: every omitted term is
        fermion-even with support disjoint from every monomial of op.
        """
        lat = self.lat
        supp = op.support()
        if supp:
            self._check_interior(supp)
        sites = lat.sites_of(supp)
        H = Operator.zero()
        for key in self._term_keys_touching(sites):
            H = H + self._term(key)
        return H

    def L(self, op: Operator) -> Operator:
        """[H, op] in the thermodynamic limit, with per-monomial locality.

        For each monomial m, only the local terms touching sites(m)
        contribute: fermion-even terms on disjoint support commute with m
        exactly (not just after cancellation between the two orderings), so
        skipping them is exact and avoids the dominant cancellation waste.

        Boundary guard: every site of supp(op) must lie in [1, W-2]^2 so
        that all touching bonds exist inside the window.
        """
        from algebra import mono_mul, mono_support
        lat = self.lat
        supp = op.support()
        if not supp:
            return Operator.zero()
        self._check_interior(supp)
        # NOTE: stays on the UNCACHED mono_mul deliberately — measured hit
        # rate in this workload is ~13%, so caching costs more than it saves
        # and evicts assembly-workload entries (~85% hit rate) from the
        # shared bounded cache.
        cache = self._term_cache
        acc: dict = {}
        for m, v in op.terms.items():
            for key in self._term_keys_touching(lat.sites_of(mono_support(m))):
                h = cache.get(key)
                if h is None:
                    h = self._term(key)
                    cache[key] = h
                for hm, hv in h.terms.items():
                    cval = hv * v
                    for k2, s in mono_mul(hm, m).items():
                        w = acc.get(k2, 0) + cval * s
                        if w:
                            acc[k2] = w
                        elif k2 in acc:
                            del acc[k2]
                    for k2, s in mono_mul(m, hm).items():
                        w = acc.get(k2, 0) - cval * s
                        if w:
                            acc[k2] = w
                        elif k2 in acc:
                            del acc[k2]
        return Operator(acc)

    def L_iter(self, op: Operator, k: int):
        """Yield op, L(op), L^2(op), ..., L^k(op)."""
        cur = op
        yield cur
        for _ in range(k):
            cur = self.L(cur)
            yield cur
