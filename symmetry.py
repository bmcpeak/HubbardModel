"""D4 x Z2(spin-flip) symmetry machinery for the Hubbard bootstrap.

Two distinct group actions live here; do not conflate them.

OPERATOR layer — ``PointGroup``: the point group about a fixed center
(possibly half-integer, e.g. (3/2, 3/2) for an even patch), acting on
operators in absolute window coordinates via mode permutations. Used to
build symmetry-adapted bases. An orbit leaving the window is a hard error.

MOMENT layer — ``canonical_moment``: the full symmetry group of the
translation-invariant state — translations x D4 x spin-flip x hermitian
conjugation (the ground state of the real Hubbard H can be chosen real,
so <M†> = <M>) — acting coordinate-freely. Reduces a monomial to a unique
canonical key with a sign, or flags the moment as vanishing identically:
if the orbit reaches the same canonical key with both signs, <m> = -<m> = 0.

Irrep conventions (D4, about the chosen center):
  classes:  {e}, {r2}, {r, r3}, {sx, sy} (axis reflections), {sd, sd'}
  A1: 1  1  1  1  1
  A2: 1  1  1 -1 -1
  B1: 1  1 -1  1 -1
  B2: 1  1 -1 -1  1
  E : 2 -2  0  0  0     (the vector rep (x, y) IS E; D(g) = the 2x2 matrix)

Row projectors: P_ab = (d/|G|) sum_g D(g)_ab rho(g)  (real rep, no
conjugation), satisfying rho(g) P_ab = sum_c D(g)_ca P_cb and
P_ab P_cd = delta_bc P_ad. Symmetry-adapted vectors are built from column
b=1: v_a = P_a1 O transforms as rho(g) v_a = sum_c D(g)_ca v_c. For the
bootstrap we keep row 1 only; on a G-closed span the row-2 block is an
exact copy.

With spin flip the group is D4 x Z2 (order 16) and irreps carry a flip
parity. Grades with dQ_up != dQ_dn are not flip-invariant: flip maps the
sector onto its mirror grade, so ops in (qu < qd) grades are FOLDED into
the mirror at intake and only one grade per mirror pair appears in the SDP.

ARCHITECTURE NOTE (SU(2)): the flip parity is a stopgap label. The spin
flip is (up to grade-dependent phases) the pi-rotation exp(i pi S_x), so
once operators carry total-spin labels S, flip parity is determined —
(-1)^S on Sz = 0 multiplets — not independent. The su2 layer will emit
blocks labeled (charges, Gamma_D4, S) through this same decompose()
output type, absorbing the ± tags.

All projector arithmetic is exact (Fraction); coefficients stay exact all
the way to the SDP layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from algebra import Operator, mono_support, bits_ascending
from modes import Lattice


# ----------------------------------------------------------------------
# D4 elements as integer 2x2 matrices, in fixed order
# ----------------------------------------------------------------------

D4_NAMES = ("e", "r", "r2", "r3", "sx", "sy", "sd", "sd2")

D4_MATS = {
    "e":   ((1, 0), (0, 1)),
    "r":   ((0, -1), (1, 0)),      # 90 deg CCW
    "r2":  ((-1, 0), (0, -1)),
    "r3":  ((0, 1), (-1, 0)),
    "sx":  ((1, 0), (0, -1)),      # (x, y) -> (x, -y)
    "sy":  ((-1, 0), (0, 1)),
    "sd":  ((0, 1), (1, 0)),       # (x, y) -> (y, x)
    "sd2": ((0, -1), (-1, 0)),
}

D4_CLASS = {"e": 0, "r2": 1, "r": 2, "r3": 2, "sx": 3, "sy": 3, "sd": 4, "sd2": 4}

# characters by class order: e, r2, {r,r3}, sigma_v, sigma_d
D4_CHARACTERS = {
    "A1": (1, 1, 1, 1, 1),
    "A2": (1, 1, 1, -1, -1),
    "B1": (1, 1, -1, 1, -1),
    "B2": (1, 1, -1, -1, 1),
    "E":  (2, -2, 0, 0, 0),
}

ONE_DIM_IRREPS = ("A1", "A2", "B1", "B2")


def _mat_apply(M, x, y):
    return (M[0][0] * x + M[0][1] * y, M[1][0] * x + M[1][1] * y)


# ----------------------------------------------------------------------
# operator-layer point group about a fixed center
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class GroupElement:
    name: str          # e.g. "r", "sx*F"
    d4: str            # D4 part
    flip: bool         # spin flip


class PointGroup:
    """D4 (x spin flip) about ``center``, acting inside the window.

    center: (cx, cy) as numbers or Fractions; half-integers allowed.
    Mode permutation tables are precomputed once; an image outside the
    window is recorded as -1 and raises only if actually used.
    """

    def __init__(self, lat: Lattice, center, include_flip: bool = True):
        self.lat = lat
        self.cx = Fraction(center[0])
        self.cy = Fraction(center[1])
        self.include_flip = include_flip
        self.elements = [GroupElement(n + ("*F" if f else ""), n, f)
                         for n in D4_NAMES
                         for f in ((False, True) if include_flip else (False,))]
        self._tables = {g: self._build_table(g) for g in self.elements}
        self._mono_cache = {}   # (m, g) -> (m', sign); projection revisits
                                # the same monomials constantly

    def _build_table(self, g: GroupElement):
        lat, M = self.lat, D4_MATS[g.d4]
        table = [-1] * lat.n_modes
        for mode in range(lat.n_modes):
            x, y, s = lat.decode(mode)
            rx, ry = _mat_apply(M, Fraction(x) - self.cx, Fraction(y) - self.cy)
            nx, ny = self.cx + rx, self.cy + ry
            if nx.denominator != 1 or ny.denominator != 1:
                raise ValueError(
                    f"center ({self.cx},{self.cy}) maps site ({x},{y}) to "
                    f"non-integer ({nx},{ny}); use integer or half-integer "
                    f"center consistent with the lattice")
            nx, ny = int(nx), int(ny)
            ns = (1 - s) if g.flip else s
            if lat.in_window(nx, ny):
                table[mode] = lat.mode(nx, ny, ns)
        return table

    def apply_mono(self, m, g: GroupElement):
        """(m', sign) for the action of g on a canonical monomial. Cached."""
        hit = self._mono_cache.get((m, g))
        if hit is not None:
            return hit
        table = self._tables[g]

        def perm(i):
            j = table[i]
            if j < 0:
                x, y, s = self.lat.decode(i)
                raise ValueError(
                    f"element {g.name} maps mode ({x},{y},{s}) outside the "
                    f"window; enlarge W or move the basis center")
            return j
        out = self.lat.permute_mono(m, perm)
        self._mono_cache[(m, g)] = out
        return out

    def clear_mono_cache(self):
        self._mono_cache.clear()

    def apply(self, op: Operator, g: GroupElement) -> Operator:
        out: dict = {}
        for m, v in op.terms.items():
            m2, s = self.apply_mono(m, g)
            out[m2] = out.get(m2, 0) + s * v
        return Operator(out)

    # -- irrep projectors --------------------------------------------------

    def _chi(self, irrep: str, g: GroupElement, flip_parity: int) -> int:
        chi = D4_CHARACTERS[irrep][D4_CLASS[g.d4]]
        if self.include_flip and g.flip and flip_parity < 0:
            chi = -chi
        return chi

    def project_1d(self, op: Operator, irrep: str, flip_parity: int = +1) -> Operator:
        """P_lambda op, exact, for a 1-dim irrep (A1/A2/B1/B2, flip parity ±)."""
        if irrep not in ONE_DIM_IRREPS:
            raise ValueError(f"unknown 1-dim irrep {irrep!r}")
        if flip_parity not in (+1, -1):
            raise ValueError(f"flip_parity must be +1 or -1, got {flip_parity!r}")
        n = len(self.elements)
        acc = Operator.zero()
        for g in self.elements:
            chi = self._chi(irrep, g, flip_parity)
            if chi:
                acc = acc + chi * self.apply(op, g)
        return acc.scale(Fraction(1, n))

    def project_E(self, op: Operator, a: int, b: int, flip_parity: int = +1) -> Operator:
        """P^E_ab op with D(g) the vector-rep matrix of the D4 part."""
        if a not in (0, 1) or b not in (0, 1):
            raise ValueError(f"E row/col indices must be 0 or 1, got ({a},{b})")
        if flip_parity not in (+1, -1):
            raise ValueError(f"flip_parity must be +1 or -1, got {flip_parity!r}")
        n = len(self.elements)
        acc = Operator.zero()
        for g in self.elements:
            d = D4_MATS[g.d4][a][b]
            if self.include_flip and g.flip and flip_parity < 0:
                d = -d
            if d:
                acc = acc + d * self.apply(op, g)
        return acc.scale(Fraction(2, n))

    def close_span(self, ops):
        """G-closure of the span: independent subset of {g . O_i}, exact."""
        red = _ExactReducer()
        out = []
        for op in ops:
            for g in self.elements:
                img = self.apply(op, g)
                if red.add(img):
                    out.append(img)
        return out


# ----------------------------------------------------------------------
# exact linear-independence reduction over monomial keys
# ----------------------------------------------------------------------

class _ExactReducer:
    """Incremental row reduction over Q, rows indexed by monomial keys.

    add(op) returns True (and absorbs op) iff op is independent of what
    has been added so far. Pivot rows are stored fully reduced.
    """

    def __init__(self):
        self.pivots = {}   # pivot monomial key -> {mono: Fraction} row

    @staticmethod
    def _rowify(op: Operator):
        return {m: Fraction(v) for m, v in op.terms.items()}

    def _reduce(self, row):
        for pkey in sorted(row.keys(), reverse=True):
            if pkey not in row:
                continue
            piv = self.pivots.get(pkey)
            if piv is None:
                continue
            f = row[pkey]
            for k, v in piv.items():
                w = row.get(k, 0) - f * v
                if w:
                    row[k] = w
                elif k in row:
                    del row[k]
        return row

    def add(self, op: Operator) -> bool:
        row = self._reduce(self._rowify(op))
        if not row:
            return False
        pkey = max(row.keys())
        inv = 1 / row[pkey]
        row = {k: v * inv for k, v in row.items()}
        for prow in self.pivots.values():
            f = prow.get(pkey)
            if f:
                for k, v in row.items():
                    w = prow.get(k, 0) - f * v
                    if w:
                        prow[k] = w
                    elif k in prow:
                        del prow[k]
        self.pivots[pkey] = row
        return True


# ----------------------------------------------------------------------
# block decomposition
# ----------------------------------------------------------------------

@dataclass
class Block:
    charges: tuple          # (dQ_up, dQ_dn)
    irrep: str              # "A1+", "B2-", "E+", ... ("+/-" = flip parity; absent if no flip)
    ops: list = field(default_factory=list)
    # For monomial bases (orbit decomposition): seeds[i] = (m_i, beta_i) with
    # ops[i] = beta_i * (P m_i), P the block's projector. Enables the
    # one-sided assembly fast path <op_i† op_j> = beta_i <m_i† op_j> (using
    # P† P = P). None for general bases.
    seeds: list = None

    def __repr__(self):
        return f"Block(charges={self.charges}, irrep={self.irrep}, n={len(self.ops)})"


def normalize_int(op: Operator):
    """Rescale to primitive integer coefficients with a fixed sign convention.

    Returns (op_int, beta) with op_int = beta * op, beta rational. Pure
    basis rescaling: PSD structure unchanged, exact arithmetic cheaper,
    operators readable.
    """
    from math import gcd
    fracs = {m: Fraction(v) for m, v in op.terms.items()}
    L = 1
    for v in fracs.values():
        d = v.denominator
        L = L * d // gcd(L, d)
    ints = {m: int(v * L) for m, v in fracs.items()}
    g = 0
    for v in ints.values():
        g = gcd(g, abs(v))
    g = g or 1
    beta = Fraction(L, g)
    ints = {m: v // g for m, v in ints.items()}
    if ints[min(ints)] < 0:
        ints = {m: -v for m, v in ints.items()}
        beta = -beta
    return Operator(ints), beta


def _split_sectors(pg: PointGroup, ops):
    """Grade ops and fold (qu < qd) grades into their mirrors."""
    lat = pg.lat
    flip_el = None
    if pg.include_flip:
        flip_el = next(g for g in pg.elements if g.d4 == "e" and g.flip)
    sectors: dict = {}
    for op in ops:
        if not op:
            continue
        grade = lat.operator_charges(op)
        if flip_el is not None and grade[0] < grade[1]:
            # FOLD, don't skip: an op in a (qu < qd) grade is flipped into
            # its mirror grade and merged with any native ops there. The
            # mirror SDP block is entry-for-entry identical (flip-invariant
            # state), so folding preserves every constraint with no
            # duplicate blocks — and no sector is lost when a trial basis
            # populates only one side of a mirror pair.
            op = pg.apply(op, flip_el)
            grade = (grade[1], grade[0])
        sectors.setdefault(grade, []).append(op)
    return sectors


def _sector_group(pg: PointGroup, grade):
    qu, qd = grade
    if qu == qd and pg.include_flip:
        return pg
    return PointGroup(pg.lat, (pg.cx, pg.cy), include_flip=False)


def _decompose_general(sub: PointGroup, grade, sops, close: bool):
    """Projector + exact-reducer path for arbitrary Operator spans."""
    if close:
        sops = sub.close_span(sops)
    blocks = []
    parities = (+1, -1) if sub.include_flip else (+1,)
    for fp in parities:
        tag = {1: "+", -1: "-"}[fp] if sub.include_flip else ""
        for irrep in ONE_DIM_IRREPS:
            red = _ExactReducer()
            blk = Block(grade, irrep + tag)
            for op in sops:
                p = sub.project_1d(op, irrep, fp)
                if p and red.add(p):
                    blk.ops.append(p)
            if blk.ops:
                blocks.append(blk)
        red = _ExactReducer()
        blk = Block(grade, "E" + tag)
        for op in sops:
            p = sub.project_E(op, 0, 0, fp)   # row 1 only
            if p and red.add(p):
                blk.ops.append(p)
        if blk.ops:
            blocks.append(blk)
    return blocks


def _decompose_monomials(sub: PointGroup, grade, monos):
    """Orbit path for monomial bases: no global row reduction.

    Orbits partition the monomial set, so vectors from different orbits are
    independent automatically. Within one orbit: a 1-dim irrep has
    multiplicity <= 1 (Frobenius reciprocity for a rep induced from the
    1-dim signed character of the stabilizer), so the first nonzero
    projection IS the basis vector; E has multiplicity <= 2, found with a
    per-orbit reducer.
    """
    parities = (+1, -1) if sub.include_flip else (+1,)
    labels = []
    for fp in parities:
        tag = {1: "+", -1: "-"}[fp] if sub.include_flip else ""
        for irrep in ONE_DIM_IRREPS:
            labels.append((irrep + tag, irrep, fp, False))
        labels.append(("E" + tag, "E", fp, True))
    collect = {lab[0]: ([], []) for lab in labels}   # label -> (ops, seeds)

    seen = set()
    for m in monos:
        if m in seen:
            continue
        orbit = []
        for g in sub.elements:
            m2, _ = sub.apply_mono(m, g)
            if m2 not in seen:
                seen.add(m2)
                orbit.append(m2)
        for label, irrep, fp, is_E in labels:
            ops_l, seeds_l = collect[label]
            if not is_E:
                for mm in orbit:
                    p = sub.project_1d(Operator({mm: 1}), irrep, fp)
                    if p:
                        q, beta = normalize_int(p)
                        ops_l.append(q)
                        seeds_l.append((mm, beta))
                        break                     # multiplicity <= 1
            else:
                red = _ExactReducer()
                found = 0
                for mm in orbit:
                    p = sub.project_E(Operator({mm: 1}), 0, 0, fp)
                    if p and red.add(p):
                        q, beta = normalize_int(p)
                        ops_l.append(q)
                        seeds_l.append((mm, beta))
                        found += 1
                        if found == 2:            # multiplicity <= 2
                            break
    blocks = []
    for label, _, _, _ in labels:
        ops_l, seeds_l = collect[label]
        if ops_l:
            blocks.append(Block(grade, label, ops=ops_l, seeds=seeds_l))
    return blocks


def decompose(pg: PointGroup, ops, close: bool = True):
    """Split a list of graded Operators into symmetry-adapted blocks.

    Monomial bases (every op a single monomial) take the fast orbit path,
    which also records (seed, beta) provenance enabling the one-sided
    assembly fast path. General spans (Krylov linear combinations) take
    the projector + exact-reducer path; close=True closes the span under
    the group first, required there for row-1-only E treatment to be
    lossless. Symmetry-adapted Krylov bases may use close=False.
    """
    sectors = _split_sectors(pg, ops)
    blocks = []
    for grade in sorted(sectors.keys()):
        sops = sectors[grade]
        sub = _sector_group(pg, grade)
        if all(len(op) == 1 for op in sops):
            monos = [next(iter(op.terms)) for op in sops]
            blocks.extend(_decompose_monomials(sub, grade, monos))
        else:
            blocks.extend(_decompose_general(sub, grade, sops, close))
    return blocks


def audit_decomposition(pg: PointGroup, ops, blocks, verbose=True):
    """Verify sum over blocks of dim(Gamma) * multiplicity == folded raw
    dimension, per grade (monomial bases). The safeguard against silently
    dropping an irrep row or a spin-flip mirror."""
    lat = pg.lat
    folded: dict = {}
    for op in ops:
        if not op or len(op) != 1:
            raise ValueError("audit supports monomial bases")
        m = next(iter(op.terms))
        g = lat.charges(m)
        if pg.include_flip and g[0] < g[1]:
            flip_el = next(e for e in pg.elements if e.d4 == "e" and e.flip)
            m, _ = pg.apply_mono(m, flip_el)
            g = (g[1], g[0])
        folded.setdefault(g, set()).add(m)
    ok = True
    lines = []
    for grade in sorted(folded.keys()):
        want = len(folded[grade])
        have = sum((2 if b.irrep.startswith("E") else 1) * len(b.ops)
                   for b in blocks if b.charges == grade)
        lines.append(f"    grade {grade}: folded dim {want}, "
                     f"sum dim*mult {have} {'OK' if want == have else 'MISMATCH'}")
        ok = ok and (want == have)
    if verbose:
        print("\n".join(lines))
    if not ok:
        raise AssertionError("irrep dimension accounting failed:\n" + "\n".join(lines))
    return True


# ----------------------------------------------------------------------
# moment-layer canonicalization (translations x D4 x flip x dagger)
# ----------------------------------------------------------------------

def _free_transform(lat: Lattice, m, mat, flip: bool):
    """Apply a D4 matrix (about the origin) + optional flip, coordinate-free.

    Composes with the translation putting the min corner at (0,0), so the
    center of rotation is irrelevant. Returns (m', sign).
    """
    per_mask = []
    coords = []
    for mask in m:
        entry = []
        for i in bits_ascending(mask):
            x, y, s = lat.decode(i)
            nx, ny = _mat_apply(mat, x, y)
            ns = (1 - s) if flip else s
            entry.append((nx, ny, ns))
            coords.append((nx, ny))
        per_mask.append(entry)
    if not coords:
        return m, 1
    minx = min(c[0] for c in coords)
    miny = min(c[1] for c in coords)
    sign = 1
    out_masks = []
    for entry in per_mask:
        images = [lat.mode(nx - minx, ny - miny, ns) for (nx, ny, ns) in entry]
        mask2 = 0
        inv = 0
        for a in range(len(images)):
            mask2 |= 1 << images[a]
            for b in range(a + 1, len(images)):
                if images[b] < images[a]:
                    inv += 1
        if inv & 1:
            sign = -sign
        out_masks.append(mask2)
    return (out_masks[0], out_masks[1]), sign


def canonical_moment(lat: Lattice, m, use_flip: bool = True,
                     use_dagger: bool = True):
    """Reduce <m> to its canonical representative under the state's symmetry.

    Returns (m_canonical, sign) with <m> = sign * <m_canonical>, or sign 0
    if the moment vanishes identically (its orbit reaches the canonical key
    with both signs). Assumes the state is translation-invariant,
    D4-invariant, spin-flip invariant, and real (dagger identification).

    The relevant symmetry facts: translations are signless; hermitian
    conjugation (dag, c) -> (c, dag) is signless in our convention; only
    the point-group permutation carries signs.
    """
    if m == (0, 0):
        return m, 1
    nbits = lat.n_modes
    best_key = None
    best_m = None
    signs_at_best = set()
    daggers = (False, True) if use_dagger else (False,)
    flips = (False, True) if use_flip else (False,)
    for dg in daggers:
        m0 = (m[1], m[0]) if dg else m
        for name in D4_NAMES:
            mat = D4_MATS[name]
            for fl in flips:
                m2, s = _free_transform(lat, m0, mat, fl)
                key = (m2[0] << nbits) | m2[1]
                if best_key is None or key < best_key:
                    best_key, best_m, signs_at_best = key, m2, {s}
                elif key == best_key:
                    signs_at_best.add(s)
    if len(signs_at_best) == 2:
        return best_m, 0
    return best_m, signs_at_best.pop()
