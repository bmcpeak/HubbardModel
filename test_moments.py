"""Tests for moments.py, plus the strongest available check of the whole
moment pipeline: validation against an exact correlated state.

The Torus2x2 ground mixture (validated to 14 digits against the
independent oracle in test_oracle_2x2.py) possesses every symmetry
canonical_moment uses — torus translations, D4 about a site, spin flip,
reality — and the free-space identifications made by canonical_moment are
a subset of torus symmetries for window-fitting monomials (min-corner
anchoring never wraps). So every identification the moment layer asserts
must hold there:

    <m> == sign * <m_canonical>          (full canonicalization)
    <m> == s_g * <g . m>                 (each transformation separately)
    <m> == <m†>                          (dagger identification)
    sign == 0  =>  <m> == 0              (symmetry kills)
    charges != 0  =>  <m> == 0           (selection rule; automatic at fixed N)

Run: python tests/test_moments.py
"""

import random
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fractions import Fraction

from algebra import Operator, dop, cop, nop
from modes import Lattice
from symmetry import canonical_moment, D4_NAMES, D4_MATS, _free_transform
from moments import MomentTable
from test_oracle_2x2 import Torus2x2

random.seed(20260713)


def rand_mono(lat, deg_up=1, deg_dn=1, balanced=True):
    """Random monomial; balanced=True forces grade (0,0)."""
    up = [lat.mode(x, y, 0) for x in range(lat.W) for y in range(lat.W)]
    dn = [lat.mode(x, y, 1) for x in range(lat.W) for y in range(lat.W)]
    dag = 0
    c = 0
    for _ in range(deg_up):
        dag |= 1 << random.choice(up)
        c |= 1 << random.choice(up) if balanced else 0
    for _ in range(deg_dn):
        dag |= 1 << random.choice(dn)
        c |= 1 << random.choice(dn) if balanced else 0
    return (dag, c)


# ----------------------------------------------------------------------
# table mechanics
# ----------------------------------------------------------------------

def test_identity_and_selection_rule():
    lat = Lattice(5)
    tab = MomentTable(lat)
    assert tab.lookup((0, 0)) == (tab.identity_id, 1)
    # charged monomial: exact zero, not a variable
    m = (1 << lat.mode(1, 1, 0), 0)
    assert tab.lookup(m) == (None, 0)
    assert len(tab) == 1  # nothing interned beyond identity


def test_symmetry_related_monomials_share_id():
    lat = Lattice(7)
    tab = MomentTable(lat)
    hits = 0
    for _ in range(200):
        m = rand_mono(lat, 1, 1)
        mid, s = tab.lookup(m)
        if mid is None:
            continue
        hits += 1
        # a random symmetry image must resolve to the same id with the
        # composed sign
        m0 = (m[1], m[0]) if random.random() < 0.5 else m
        m2, sg = _free_transform(lat, m0, D4_MATS[random.choice(D4_NAMES)],
                                 random.random() < 0.5)
        mid2, s2 = tab.lookup(m2)
        assert mid2 == mid and s == sg * s2
    assert hits > 50


def test_resolve_linearity_and_dagger():
    lat = Lattice(5)
    tab = MomentTable(lat)
    for _ in range(50):
        A = Operator({rand_mono(lat): random.choice((-2, -1, 1, 3)),
                      rand_mono(lat): random.choice((-1, 1))})
        B = Operator({rand_mono(lat): random.choice((-1, 1, 2))})
        rA, rB = tab.resolve(A), tab.resolve(B)
        rAB = tab.resolve(A + 2 * B)
        for k in set(rA) | set(rB) | set(rAB):
            assert rAB.get(k, 0) == rA.get(k, 0) + 2 * rB.get(k, 0)
        # dagger identification: <O†> = <O> for real coefficients
        assert tab.resolve(A.dag()) == rA


def test_exactness_preserved():
    lat = Lattice(5)
    tab = MomentTable(lat)
    op = Operator({rand_mono(lat): Fraction(1, 3), rand_mono(lat): 2})
    row = tab.resolve(op)
    assert all(isinstance(v, (int, Fraction)) for v in row.values())


def test_memo_and_drop():
    lat = Lattice(5)
    tab = MomentTable(lat)
    ms = [rand_mono(lat) for _ in range(100)]
    for m in ms:
        tab.lookup(m)
    n1 = len(tab)
    raw = tab.stats()["raw_memo"]
    assert raw >= len(set(ms))
    dropped = tab.drop_raw_memo()
    assert dropped == raw
    # lookups still correct after dropping the memo
    for m in ms[:20]:
        mid, s = tab.lookup(m)
    assert len(tab) == n1, "dropping the memo must not create new moment ids"


# ----------------------------------------------------------------------
# state-level validation on the exact torus ground mixture
# ----------------------------------------------------------------------

def test_moment_pipeline_against_exact_state():
    tor = Torus2x2()
    lat = tor.lat
    tab = MomentTable(lat)

    def ev(m):
        return tor.expval(Operator({m: 1}))

    checked_ids = 0
    checked_kills = 0
    for _ in range(150):
        deg = random.choice(((1, 1), (2, 0), (1, 2), (2, 2)))
        m = rand_mono(lat, *deg)
        if m == (0, 0):
            continue
        vm = ev(m)

        # every individual transformation: <m> = s_g <g.m>, <m> = <m†>
        for name in D4_NAMES:
            for fl in (False, True):
                m2, sg = _free_transform(lat, m, D4_MATS[name], fl)
                assert abs(vm - sg * ev(m2)) < 1e-9, \
                    f"<m> != s<g.m> at {name}, flip={fl}"
        assert abs(vm - ev((m[1], m[0]))) < 1e-9, "<m> != <m†>"

        # full pipeline through the table
        mid, s = tab.lookup(m)
        if mid is None:
            assert abs(vm) < 1e-9, "moment flagged zero is nonzero in state"
            checked_kills += 1
        else:
            assert abs(vm - s * ev(tab.monos[mid])) < 1e-9, \
                "<m> != sign * <m_canonical>"
            checked_ids += 1
    assert checked_ids > 30
    print(f"    state-level: {checked_ids} canonicalizations, "
          f"{checked_kills} exact zeros verified against torus GS")


def test_resolve_against_exact_state():
    """<O> computed via resolve + canonical values == direct <O>."""
    tor = Torus2x2()
    lat = tor.lat
    tab = MomentTable(lat)
    for _ in range(30):
        op = Operator({rand_mono(lat, 1, 1): random.choice((-2, 1, 3)),
                       rand_mono(lat, 2, 0): random.choice((-1, 1)),
                       rand_mono(lat, 1, 2): random.choice((-1, 2))})
        direct = tor.expval(op)
        row = tab.resolve(op)
        via_table = sum(float(v) * tor.expval(Operator({tab.monos[k]: 1}))
                        for k, v in row.items())
        assert abs(direct - via_table) < 1e-9, "resolve row disagrees with state"


def test_mirror_folding_regression():
    """decompose must not lose a sector when only one mirror grade is given."""
    from symmetry import PointGroup, decompose
    lat = Lattice(5)
    pg = PointGroup(lat, (2, 2), include_flip=True)
    # ops ONLY in grade (0,1): single down-spin daggers
    ops = [Operator({(1 << lat.mode(x, y, 1), 0): 1})
           for x in range(1, 4) for y in range(1, 4)]
    blocks = decompose(pg, ops)
    total = sum(len(b.ops) for b in blocks)
    assert total > 0, "mirror-only basis was silently discarded"
    assert all(b.charges == (1, 0) for b in blocks), "fold target wrong"
    assert total == 7  # 9 site ops -> 3 A1 + 1 B1 + 1 B2 + 2 E(row 1)



def test_exhaustive_2x2_moment_regression():
    """Every neutral monomial up to degree 4 on the 2x2 torus:
    (a) all raw monomials assigned one moment ID agree (up to sign) with
        the ID's canonical value in the exact ground mixture;
    (b) every symmetry-killed or selection-rule-zero monomial is
        numerically zero. Exhaustive version of the sampled test above."""
    tor = Torus2x2()
    lat = tor.lat
    tab = MomentTable(lat)
    by_id = {}
    kills = 0
    checked = 0
    for dag in range(256):
        for c in range(256):
            m = (dag, c)
            deg = bin(dag).count("1") + bin(c).count("1")
            if deg == 0 or deg > 4:
                continue
            mid, s = tab.lookup(m)
            v = tor.expval(Operator({m: 1}))
            if mid is None:
                assert abs(v) < 1e-9, f"flagged-zero moment nonzero: {m}, {v}"
                kills += 1
                continue
            checked += 1
            ref = by_id.setdefault(mid, s * v)      # canonical value
            assert abs(s * v - ref) < 1e-9, \
                f"raw monomials of moment {mid} disagree: {s*v} vs {ref}"
    print(f"    exhaustive 2x2: {checked} neutral monomials over "
          f"{len(by_id)} moment ids, {kills} exact zeros — all consistent")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"ok  {f.__name__}")
    print(f"\nall {len(fns)} tests passed")
