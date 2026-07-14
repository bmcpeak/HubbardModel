"""Lattice geometry for the square-lattice Hubbard bootstrap.

Encodes the embedding of abstract algebra modes (algebra.py) into a finite
W x W window of the infinite square lattice:

    mode(x, y, s) = 2*(y*W + x) + s,   x, y in [0, W),  s in {0 (up), 1 (down)}.

The window is a *scratch canvas* for the thermodynamic-limit problem, not a
finite system: there are no periodic boundaries and nothing may ever touch
the window edge in a way that would require sites outside it (hamiltonian.py
enforces this). Choose W with margin for the Krylov depth you intend to run;
masks are arbitrary-precision ints, so large W costs memory per stored
monomial (~2*W*W bits), nothing else.

Coordinate conventions:
  * OPERATOR layer (seeds, Krylov iterates, basis elements): absolute window
    coordinates. Relative placement between basis elements is physical; only
    the overall translation is a symmetry. Nothing in this layer may silently
    re-center an operator.
  * MOMENT layer (interned expectation values): translation-canonicalized via
    canonical_translate(), min corner of the support at (0, 0). Applied only
    when a monomial becomes a moment key.

Sign facts used here (all verified in tests):
  * translations add a constant to the mode index -> monotone -> sign +1;
  * a general mode permutation (D4 element, spin flip) permutes the canonical
    string out of order; re-sorting contributes the inversion parity of the
    induced permutation on each mask's set bits (dag and c independently).
"""

from __future__ import annotations

from algebra import bits_ascending, mono_support


class Lattice:
    def __init__(self, W: int):
        if W < 2:
            raise ValueError("W must be >= 2")
        self.W = W
        self.n_modes = 2 * W * W
        # mask of all spin-up modes (even bit positions) in the window
        up = 0
        for k in range(W * W):
            up |= 1 << (2 * k)
        self.UP_MASK = up
        self.DN_MASK = up << 1

    # -- encoding -------------------------------------------------------

    def mode(self, x: int, y: int, s: int) -> int:
        W = self.W
        if not (0 <= x < W and 0 <= y < W and s in (0, 1)):
            raise ValueError(f"mode ({x},{y},{s}) outside window W={W}")
        return 2 * (y * W + x) + s

    def decode(self, mode: int):
        s = mode & 1
        site = mode >> 1
        return site % self.W, site // self.W, s

    def in_window(self, x: int, y: int) -> bool:
        return 0 <= x < self.W and 0 <= y < self.W

    # -- support geometry -------------------------------------------------

    def sites_of(self, mask: int):
        """Set of (x, y) sites carrying any mode of the mask."""
        out = set()
        for i in bits_ascending(mask):
            x, y, _ = self.decode(i)
            out.add((x, y))
        return out

    def bbox(self, mask: int):
        """(minx, miny, maxx, maxy) of the support, or None for the identity."""
        if not mask:
            return None
        minx = miny = self.W
        maxx = maxy = -1
        for i in bits_ascending(mask):
            x, y, _ = self.decode(i)
            minx = min(minx, x); maxx = max(maxx, x)
            miny = min(miny, y); maxy = max(maxy, y)
        return (minx, miny, maxx, maxy)

    def mono_bbox(self, m):
        return self.bbox(mono_support(m))

    # -- translations (sign +1 always) -------------------------------------

    def translate_mask(self, mask: int, dx: int, dy: int) -> int:
        """Shift a mask by (dx, dy). Raises if any bit would leave the window.

        Implemented as a single big-int shift, which is exact because the
        x-range validity check below guarantees no bit crosses a row
        boundary.
        """
        if not mask:
            return 0
        minx, miny, maxx, maxy = self.bbox(mask)
        if not (0 <= minx + dx and maxx + dx < self.W
                and 0 <= miny + dy and maxy + dy < self.W):
            raise ValueError(
                f"translation ({dx},{dy}) pushes support bbox "
                f"({minx},{miny},{maxx},{maxy}) outside window W={self.W}")
        shift = 2 * (dy * self.W + dx)
        return mask << shift if shift >= 0 else mask >> (-shift)

    def translate_mono(self, m, dx: int, dy: int):
        return (self.translate_mask(m[0], dx, dy),
                self.translate_mask(m[1], dx, dy))

    def canonical_translate(self, m):
        """Shift so the support's min corner sits at (0, 0).

        Returns (m_canonical, (dx, dy)) with m = translate(m_canonical, -dx, -dy)
        ... i.e. dx, dy are the applied shift. MOMENT layer only.
        """
        supp = mono_support(m)
        if not supp:
            return m, (0, 0)
        minx, miny, _, _ = self.bbox(supp)
        if minx == 0 and miny == 0:
            return m, (0, 0)
        return self.translate_mono(m, -minx, -miny), (-minx, -miny)

    def shift_operator(self, op, dx: int, dy: int):
        """Translate every monomial of an Operator (for placing seed copies)."""
        from algebra import Operator
        return Operator({self.translate_mono(m, dx, dy): v
                         for m, v in op.terms.items()})

    # -- general mode permutations (D4, spin flip): signs matter -----------

    @staticmethod
    def _permute_mask_signed(mask: int, perm):
        """Apply an injective mode map to a mask.

        Returns (new_mask, sign) where sign is the inversion parity of the
        image sequence taken in ascending source order — the parity of the
        permutation that re-sorts the substituted string into canonical
        order. Valid for both the dag part (ascending convention) and the
        c part (descending convention): conjugation by reversal preserves
        parity.
        """
        images = [perm(i) for i in bits_ascending(mask)]
        new_mask = 0
        inv = 0
        for a in range(len(images)):
            ia = images[a]
            bit = 1 << ia
            if new_mask & bit:
                raise ValueError("mode map is not injective on the mask")
            new_mask |= bit
            for b in range(a + 1, len(images)):
                if images[b] < ia:
                    inv += 1
        return new_mask, (-1 if inv & 1 else 1)

    def permute_mono(self, m, perm):
        """Apply an injective mode map to a canonical monomial.

        Returns (m', sign) such that substituting a_i -> a_{perm(i)} in the
        canonical string of m equals sign * (canonical string of m').
        """
        new_dag, s1 = self._permute_mask_signed(m[0], perm)
        new_c, s2 = self._permute_mask_signed(m[1], perm)
        return (new_dag, new_c), s1 * s2

    def site_spin_perm(self, site_map, spin_map=None):
        """Build a mode map from a site map (x,y)->(x',y') and optional spin map."""
        def perm(mode):
            x, y, s = self.decode(mode)
            nx, ny = site_map(x, y)
            ns = spin_map(s) if spin_map is not None else s
            return self.mode(nx, ny, ns)
        return perm

    # -- grading ------------------------------------------------------------

    def charges(self, m):
        """(dQ_up, dQ_dn): net creation minus annihilation per spin species.

        A moment <m> vanishes unless both are zero (U(1) x U(1) selection
        rule); operators are graded by this pair, and <O_i^dag O_j> = 0
        unless the grades match.
        """
        dag, c = m
        return ((dag & self.UP_MASK).bit_count() - (c & self.UP_MASK).bit_count(),
                (dag & self.DN_MASK).bit_count() - (c & self.DN_MASK).bit_count())

    def particle_charge(self, m) -> int:
        """q = dQ_up + dQ_dn: total U(1) charge of the monomial."""
        u, d = self.charges(m)
        return u + d

    def twice_sz(self, m) -> int:
        """2m_z = dQ_up - dQ_dn.

        SU(2) relates grades of equal q and different 2m_z (S± steps 2m_z
        by ±2); the su2 adaptation layer keys on (q, 2m_z), while the
        finer (dQ_up, dQ_dn) grading remains the Gram-matrix selection
        rule. Convention: s=0 is UP throughout this codebase.
        """
        u, d = self.charges(m)
        return u - d

    def operator_charges(self, op):
        """Common grade of an Operator, or raises if mixed (bug guard)."""
        grades = {self.charges(m) for m in op.terms}
        if len(grades) > 1:
            raise ValueError(f"operator mixes charge grades: {grades}")
        return grades.pop() if grades else (0, 0)
