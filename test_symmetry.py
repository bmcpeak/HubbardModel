"""Tests for symmetry.py.

Layers of verification, weakest to strongest:
  1. group structure: mode-permutation tables form a faithful D4 x Z2;
     the signed action on monomials is a homomorphism;
  2. projector algebra: idempotence, mutual orthogonality, completeness,
     and the irrep transformation laws (including both E rows);
  3. canonical_moment: orbit consistency of key and sign; detection of
     moments that vanish identically by symmetry;
  4. end-to-end ED: exact ground state of the 3x3 OBC Hubbard model in the
     (2,2) sector; verify <g.O> = <O>, cross-irrep blocks of <O_i† O_j>
     vanish, the full moment matrix is PSD, and the two E-row blocks are
     equal. The Fock-space applier used for ED is validated against
     mono_mul first, so it is an independent check of the operator algebra
     as well.

Run: python tests/test_symmetry.py
"""

import itertools
import random
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from algebra import Operator, bits_ascending, bits_descending, mono_mul, parity
from modes import Lattice
from hamiltonian import Hubbard
from symmetry import (PointGroup, GroupElement, decompose, canonical_moment,
                      D4_NAMES, D4_MATS, ONE_DIM_IRREPS, _free_transform)

random.seed(20260713)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def mat_mul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(2)) for j in range(2))
                 for i in range(2))


def d4_product(n1, n2):
    """Name of the D4 element with matrix M(n1) @ M(n2)."""
    M = mat_mul(D4_MATS[n1], D4_MATS[n2])
    for name, N in D4_MATS.items():
        if N == M:
            return name
    raise AssertionError("D4 not closed?!")


def rand_interior_op(lat, lo, hi, nterms=3, deg=2):
    """Random operator supported on [lo, hi]^2 (site coords)."""
    modes = [lat.mode(x, y, s) for x in range(lo, hi + 1)
             for y in range(lo, hi + 1) for s in (0, 1)]
    t = {}
    for _ in range(nterms):
        dag = 0
        c = 0
        for _ in range(deg):
            if random.random() < 0.5:
                dag |= 1 << random.choice(modes)
            else:
                c |= 1 << random.choice(modes)
        t[(dag, c)] = random.choice((-2, -1, 1, 2))
    return Operator(t)


# ----------------------------------------------------------------------
# 1. group structure
# ----------------------------------------------------------------------

def test_group_closure_and_homomorphism():
    lat = Lattice(7)
    pg = PointGroup(lat, (3, 3), include_flip=True)
    by_key = {(g.d4, g.flip): g for g in pg.elements}
    for g in pg.elements:
        for h in pg.elements:
            gh = by_key[(d4_product(g.d4, h.d4), g.flip ^ h.flip)]
            # table composition
            tg, th, tgh = pg._tables[g], pg._tables[h], pg._tables[gh]
            for i in range(lat.n_modes):
                if th[i] >= 0 and tg[th[i]] >= 0:
                    assert tg[th[i]] == tgh[i], f"table comp fail {g.name},{h.name}"
    # signed action is a homomorphism on operators
    for _ in range(40):
        op = rand_interior_op(lat, 2, 4)
        g, h = random.choice(pg.elements), random.choice(pg.elements)
        gh = by_key[(d4_product(g.d4, h.d4), g.flip ^ h.flip)]
        assert pg.apply(pg.apply(op, h), g) == pg.apply(op, gh), \
            f"signed rep fail: {g.name} o {h.name}"


def test_half_integer_center():
    """D4 about a plaquette center (1/2, 1/2) permutes the 2x2 plaquette."""
    lat = Lattice(6)
    from fractions import Fraction
    pg = PointGroup(lat, (Fraction(5, 2), Fraction(5, 2)), include_flip=False)
    r = next(g for g in pg.elements if g.name == "r")
    # site (2,2) -> (3,2) under 90deg about (2.5, 2.5): x' = cx - (y-cy) = 3, y' = cy + (x-cx) = 2
    assert pg._tables[r][lat.mode(2, 2, 0)] == lat.mode(3, 2, 0)


# ----------------------------------------------------------------------
# 2. projector algebra and transformation laws
# ----------------------------------------------------------------------

def all_projections(pg, op):
    """[(label, projected_op)] over the complete set of irreps x rows."""
    out = []
    parities = (+1, -1) if pg.include_flip else (+1,)
    for fp in parities:
        for irr in ONE_DIM_IRREPS:
            out.append((f"{irr}{fp:+d}", pg.project_1d(op, irr, fp)))
        for a in (0, 1):
            out.append((f"E{fp:+d}_row{a+1}", pg.project_E(op, a, a, fp)))
    return out


def test_projector_completeness_and_orthogonality():
    lat = Lattice(9)
    pg = PointGroup(lat, (4, 4), include_flip=True)
    for _ in range(15):
        op = rand_interior_op(lat, 2, 6)
        projs = all_projections(pg, op)
        # completeness
        assert sum(p for _, p in projs) == op, "completeness fail"
        # idempotence & orthogonality for the 1-dim irreps
        for fp in (+1, -1):
            for irr in ONE_DIM_IRREPS:
                p = pg.project_1d(op, irr, fp)
                assert pg.project_1d(p, irr, fp) == p, "idempotence fail"
                for fp2 in (+1, -1):
                    for irr2 in ONE_DIM_IRREPS:
                        if (irr2, fp2) != (irr, fp):
                            assert not pg.project_1d(p, irr2, fp2), "orthogonality fail"
        # E row projector idempotence, orthogonality to 1-dim
        p11 = pg.project_E(op, 0, 0, +1)
        assert pg.project_E(p11, 0, 0, +1) == p11
        assert not pg.project_1d(p11, "A1", +1)
        assert not pg.project_E(p11, 1, 1, +1)   # P_22 P_11 = 0


def test_transformation_laws():
    lat = Lattice(9)
    pg = PointGroup(lat, (4, 4), include_flip=True)
    for _ in range(15):
        op = rand_interior_op(lat, 3, 5)
        for fp in (+1, -1):
            # 1-dim: g.(P op) = chi(g) (P op)
            for irr in ONE_DIM_IRREPS:
                p = pg.project_1d(op, irr, fp)
                if not p:
                    continue
                for g in pg.elements:
                    chi = pg._chi(irr, g, fp)
                    assert pg.apply(p, g) == chi * p, f"{irr} law fail at {g.name}"
            # E: v_a = P_a1 op transforms as g.v_a = sum_c D(g)_ca v_c
            v = [pg.project_E(op, 0, 0, fp) + Operator.zero(), None]
            v[0] = pg.project_E(op, 0, 0, fp)
            v[1] = pg.project_E(op, 1, 0, fp)
            if not v[0] and not v[1]:
                continue
            for g in pg.elements:
                D = D4_MATS[g.d4]
                fsign = -1 if (g.flip and fp < 0) else 1
                for a in (0, 1):
                    lhs = pg.apply(v[a], g)
                    rhs = fsign * (D[0][a] * v[0] + D[1][a] * v[1])
                    assert lhs == rhs, f"E law fail at {g.name}, row {a+1}"


# ----------------------------------------------------------------------
# 3. canonical_moment
# ----------------------------------------------------------------------

def rand_mono_in(lat, lo, hi, deg=3):
    modes = [lat.mode(x, y, s) for x in range(lo, hi + 1)
             for y in range(lo, hi + 1) for s in (0, 1)]
    dag = 0
    c = 0
    for _ in range(deg):
        if random.random() < 0.5:
            dag |= 1 << random.choice(modes)
        else:
            c |= 1 << random.choice(modes)
    return (dag, c)


def test_canonical_moment_orbit_consistency():
    lat = Lattice(9)
    for _ in range(300):
        m = rand_mono_in(lat, 2, 6)
        if m == (0, 0):
            continue
        k1, s1 = canonical_moment(lat, m)
        # random symmetry image: dagger? point group (free), translation
        m0 = (m[1], m[0]) if random.random() < 0.5 else m
        mat = D4_MATS[random.choice(D4_NAMES)]
        m2, s = _free_transform(lat, m0, mat, random.random() < 0.5)
        m2 = lat.translate_mono(m2, random.randint(0, 2), random.randint(0, 2))
        k2, s2 = canonical_moment(lat, m2)
        assert k1 == k2, "orbit members canonicalize differently"
        if s1 == 0:
            assert s2 == 0, "vanishing flag inconsistent across orbit"
        else:
            assert s1 == s * s2, "sign relation <m> = s <m2> violated"


def test_vanishing_moment_detected():
    """Find a monomial killed by symmetry and verify the kill honestly."""
    lat = Lattice(9)
    found = None
    for m in [rand_mono_in(lat, 3, 5, deg=d) for d in (2, 3, 4) for _ in range(400)]:
        if m == (0, 0):
            continue
        k, s = canonical_moment(lat, m)
        if s == 0:
            found = m
            break
    assert found is not None, "no symmetry-vanishing moment found in scan"
    # honest verification: some symmetry image equals the canonical rep
    # with BOTH signs
    signs = set()
    k, _ = canonical_moment(lat, found)
    for dg in (False, True):
        m0 = (found[1], found[0]) if dg else found
        for name in D4_NAMES:
            for fl in (False, True):
                m2, s = _free_transform(lat, m0, D4_MATS[name], fl)
                if m2 == k:
                    signs.add(s)
    assert signs == {1, -1}, "flagged moment is not actually killed"


# ----------------------------------------------------------------------
# 4. end-to-end ED on the 3x3 OBC Hubbard model
# ----------------------------------------------------------------------
# Fock states are occupation masks over the 18 modes of a W=3 window.
# Convention: |f> = (prod of a†_i over i in f, ASCENDING order) |0>.
# Applying the canonical string of (dag, c): annihilators act first
# (rightmost = smallest), then daggers (rightmost = largest).

def apply_mono_state(m, f):
    """(sign, f') for  (dag,c)|f>,  or None if killed."""
    dag, c = m
    sign = 1
    for j in bits_ascending(c):
        bit = 1 << j
        if not (f & bit):
            return None
        if parity(f & (bit - 1)):
            sign = -sign
        f ^= bit
    for i in bits_descending(dag):
        bit = 1 << i
        if f & bit:
            return None
        if parity(f & (bit - 1)):
            sign = -sign
        f |= bit
    return sign, f


def test_state_applier_is_a_representation():
    """apply(m1, apply(m2, f)) == apply(mono_mul(m1, m2), f) on random data.

    This pins the applier to the SAME algebra as algebra.py, making the ED
    below a genuine check of the symmetry machinery rather than of two
    independently wrong conventions agreeing by luck.
    """
    n = 6
    for _ in range(500):
        m1 = (random.getrandbits(n), random.getrandbits(n))
        m2 = (random.getrandbits(n), random.getrandbits(n))
        f = random.getrandbits(n)
        # LHS: sequential application
        lhs = {}
        r2 = apply_mono_state(m2, f)
        if r2 is not None:
            r1 = apply_mono_state(m1, r2[1])
            if r1 is not None:
                lhs[r1[1]] = r1[0] * r2[0]
        # RHS: apply the normal-ordered product
        rhs = {}
        for key, s in mono_mul(m1, m2).items():
            r = apply_mono_state(key, f)
            if r is not None:
                rhs[r[1]] = rhs.get(r[1], 0) + s * r[0]
        rhs = {k: v for k, v in rhs.items() if v}
        assert lhs == rhs, "state applier disagrees with mono_mul"


class ED3x3:
    """Exact ground state of the 3x3 OBC Hubbard model in the (2,2) sector."""

    def __init__(self, t=1, U=8):
        self.lat = Lattice(3)
        lat = self.lat
        hub = Hubbard(lat, t=t, U=U)
        H = Operator.zero()
        for x in range(3):
            for y in range(3):
                H = H + hub.interaction(x, y)
                if x + 1 < 3:
                    H = H + hub.bond(x, y, "x")
                if y + 1 < 3:
                    H = H + hub.bond(x, y, "y")
        self.H = H
        up_modes = [lat.mode(x, y, 0) for x in range(3) for y in range(3)]
        dn_modes = [lat.mode(x, y, 1) for x in range(3) for y in range(3)]
        states = []
        for us in itertools.combinations(up_modes, 2):
            for ds in itertools.combinations(dn_modes, 2):
                states.append(sum(1 << i for i in us) + sum(1 << i for i in ds))
        self.states = states
        self.index = {f: k for k, f in enumerate(states)}
        dim = len(states)
        Hm = np.zeros((dim, dim))
        for k, f in enumerate(states):
            for m, v in H.terms.items():
                r = apply_mono_state(m, f)
                if r is not None:
                    Hm[self.index[r[1]], k] += float(v) * r[0]
        assert np.max(np.abs(Hm - Hm.T)) < 1e-12
        w, V = np.linalg.eigh(Hm)
        assert w[1] - w[0] > 1e-8, "degenerate GS; pick another sector"
        self.e0 = w[0]
        self.gs = V[:, 0]

    def apply_op(self, op: Operator):
        """op|GS> as {mask: amplitude} over the full Fock space."""
        out = {}
        for k, f in enumerate(self.states):
            a = self.gs[k]
            if abs(a) < 1e-15:
                continue
            for m, v in op.terms.items():
                r = apply_mono_state(m, f)
                if r is not None:
                    key = r[1]
                    out[key] = out.get(key, 0.0) + float(v) * r[0] * a
        return out

    def expval(self, op: Operator) -> float:
        img = self.apply_op(op)
        return sum(self.gs[self.index[f]] * a for f, a in img.items()
                   if f in self.index)


def dict_dot(d1, d2):
    if len(d2) < len(d1):
        d1, d2 = d2, d1
    return sum(v * d2.get(k, 0.0) for k, v in d1.items())


def test_ed_symmetry_invariance_and_blocks():
    ed = ED3x3()
    lat = ed.lat
    pg = PointGroup(lat, (1, 1), include_flip=True)

    # (a) <g.O> == <O> for random grade-(0,0) operators
    for _ in range(10):
        modes = list(range(lat.n_modes))
        t = {}
        for _ in range(3):
            i, j = random.choice(modes), random.choice(modes)
            if (i ^ j) & 1 == 0:   # same spin -> grade (0,0)
                t[(1 << i, 1 << j)] = random.choice((-1, 1, 2))
        op = Operator(t)
        if not op:
            continue
        e = ed.expval(op)
        for g in pg.elements:
            assert abs(ed.expval(pg.apply(op, g)) - e) < 1e-9, \
                f"<g.O> != <O> at {g.name}"

    # (b) grade (0,0), degree <= 2 basis: all d_i c_j with same spin, + identity
    ops = [Operator.identity()]
    for i in range(lat.n_modes):
        for j in range(lat.n_modes):
            if (i ^ j) & 1 == 0:
                ops.append(Operator({(1 << i, 1 << j): 1}))
    blocks = decompose(pg, ops)
    labels = [(b.irrep, len(b.ops)) for b in blocks]
    total = sum(n for _, n in labels)
    e_kept = sum(n for l, n in labels if l.startswith("E"))
    print(f"    grade (0,0) blocks: {labels}  (total {total}, from {len(ops)} ops)")
    # row 1 of E is kept, row 2 (same dimension) deliberately dropped:
    # kept + dropped-row-2 must exactly exhaust the closed span
    assert total + e_kept == len(ops), \
        "irrep decomposition lost dimensions beyond the E row-2 copies"

    # images of every projected operator
    images = {}
    for b in blocks:
        images[b.irrep] = [ed.apply_op(op) for op in b.ops]

    # cross-irrep blocks of <O_i† O_j> vanish
    scale = max(abs(dict_dot(v, v)) for vs in images.values() for v in vs) or 1.0
    for l1, vs1 in images.items():
        for l2, vs2 in images.items():
            if l1 == l2:
                continue
            worst = max(abs(dict_dot(v1, v2)) for v1 in vs1 for v2 in vs2)
            assert worst < 1e-9 * scale, \
                f"cross-block <{l1}|{l2}> = {worst:.2e}"

    # full moment matrix is PSD (it comes from an actual state)
    allv = [v for vs in images.values() for v in vs]
    n = len(allv)
    M = np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            M[a, b] = M[b, a] = dict_dot(allv[a], allv[b])
    wmin = np.linalg.eigvalsh(M)[0]
    assert wmin > -1e-9 * scale, f"moment matrix not PSD: {wmin}"

    # (c) the two E rows give identical blocks
    for fp, tag in ((+1, "E+"), (-1, "E-")):
        row1, row2 = [], []
        red = None
        from symmetry import _ExactReducer
        red1, red2 = _ExactReducer(), _ExactReducer()
        for op in ops:
            p1 = pg.project_E(op, 0, 0, fp)
            p2 = pg.project_E(op, 1, 0, fp)
            if p1 and red1.add(p1):
                row1.append(p1)
                red2.add(p2)
                row2.append(p2)
        if not row1:
            continue
        v1 = [ed.apply_op(op) for op in row1]
        v2 = [ed.apply_op(op) for op in row2]
        n = len(v1)
        M1 = np.array([[dict_dot(v1[a], v1[b]) for b in range(n)] for a in range(n)])
        M2 = np.array([[dict_dot(v2[a], v2[b]) for b in range(n)] for a in range(n)])
        assert np.max(np.abs(M1 - M2)) < 1e-9 * scale, f"{tag} row blocks differ"
    print(f"    E-row block equality verified; GS energy {ed.e0:.6f}")


def test_charged_grade_blocks():
    """Grade (1,0): 9 single-dagger ops; D4-only decomposition, ED cross-check."""
    ed = ED3x3()
    lat = ed.lat
    pg = PointGroup(lat, (1, 1), include_flip=True)
    ops = [Operator({(1 << lat.mode(x, y, 0), 0): 1})
           for x in range(3) for y in range(3)]
    blocks = decompose(pg, ops)
    total = sum(len(b.ops) for b in blocks)
    # 9 sites about a center split under D4 as 3 A1 + 1 B1 + 1 B2 + 2 E(row1 kept -> 2)
    print(f"    grade (1,0) blocks: {[(b.irrep, len(b.ops)) for b in blocks]}")
    assert total == 9 - 2, "expect 7 kept ops: 2 of 4 E components live in row 2"
    images = {b.irrep: [ed.apply_op(op) for op in b.ops] for b in blocks}
    scale = max(abs(dict_dot(v, v)) for vs in images.values() for v in vs) or 1.0
    for l1, vs1 in images.items():
        for l2, vs2 in images.items():
            if l1 != l2:
                worst = max(abs(dict_dot(u, v)) for u in vs1 for v in vs2)
                assert worst < 1e-9 * scale, f"cross-block <{l1}|{l2}>"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"ok  {f.__name__}")
    print(f"\nall {len(fns)} tests passed")
