"""Tests for modes.py and hamiltonian.py.

Promotes the earlier informal checks to a formal suite and adds the
structural identities of the Liouvillian that the Hankel construction
rests on:

    L(O†) = -L(O)†          (one leg of  M_ij = <a† L^{i+j} b>)
    L(T_a O) = T_a L(O)     (translation covariance)
    grade(L(O)) = grade(O)  (both (dQ_up, dQ_dn) and (q, 2m_z))
    supp(L(O)) grows by at most one lattice step

plus geometry: translation composition/inverse, canonical_translate
idempotence, permutation signs against direct generator-product
substitution, the exactness of the per-monomial locality filter against a
fully materialized window Hamiltonian, and the hard boundary guards.

Run: python tests/test_modes_hamiltonian.py
"""

import random
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from algebra import Operator, commutator, dop, cop
from modes import Lattice
from hamiltonian import Hubbard

random.seed(20260713)


def rand_interior_op(lat, lo, hi, nterms=3, deg=2, coeffs=(-2, -1, 1, 2)):
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
        t[(dag, c)] = random.choice(coeffs)
    return Operator(t)


# ----------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------

def test_translation_composition_and_inverse():
    lat = Lattice(9)
    for _ in range(100):
        op = rand_interior_op(lat, 3, 5)
        for m in op.terms:
            a = (random.randint(-2, 2), random.randint(-2, 2))
            b = (random.randint(-1, 1), random.randint(-1, 1))
            m1 = lat.translate_mono(lat.translate_mono(m, *a), *b)
            m2 = lat.translate_mono(m, a[0] + b[0], a[1] + b[1])
            assert m1 == m2, "translation composition fail"
            assert lat.translate_mono(lat.translate_mono(m, *a), -a[0], -a[1]) == m


def test_canonical_translate_idempotent():
    lat = Lattice(9)
    for _ in range(200):
        m = next(iter(rand_interior_op(lat, 2, 7, nterms=1, deg=3).terms), None)
        if m is None:
            continue
        mc, off = lat.canonical_translate(m)
        mc2, off2 = lat.canonical_translate(mc)
        assert mc2 == mc and off2 == (0, 0), "canonical_translate not idempotent"
        # and the offset actually inverts
        assert lat.translate_mono(mc, -off[0], -off[1]) == m


def test_permutation_signs_via_generator_products():
    """permute_mono sign == direct substitution a_i -> a_{perm(i)} in the string."""
    lat = Lattice(4)
    n = lat.n_modes
    for _ in range(300):
        dags = sorted(random.sample(range(n), random.randint(0, 4)))
        cs = sorted(random.sample(range(n), random.randint(0, 4)), reverse=True)
        dag_mask = sum(1 << i for i in dags)
        c_mask = sum(1 << j for j in cs)
        table = list(range(n))
        random.shuffle(table)
        m2, sign = lat.permute_mono((dag_mask, c_mask), lambda i: table[i])
        direct = Operator.identity()
        for i in dags:
            direct = direct * dop(table[i])
        for j in cs:
            direct = direct * cop(table[j])
        assert direct == Operator({m2: sign}), "permutation sign mismatch"


def test_charge_helpers():
    lat = Lattice(5)
    m = (1 << lat.mode(1, 1, 0) | 1 << lat.mode(2, 1, 0),   # two up daggers
         1 << lat.mode(1, 2, 1))                            # one down annihilator
    assert lat.charges(m) == (2, -1)
    assert lat.particle_charge(m) == 1
    assert lat.twice_sz(m) == 3


# ----------------------------------------------------------------------
# Liouvillian identities
# ----------------------------------------------------------------------

def test_L_dagger_antisymmetry():
    """L(O†) == -L(O)†, exactly. The Hankel identity's first leg."""
    lat = Lattice(9)
    hub = Hubbard(lat)
    for _ in range(30):
        op = rand_interior_op(lat, 3, 5)
        assert hub.L(op.dag()) == -(hub.L(op).dag()), "L-dagger antisymmetry fail"


def test_L_translation_covariance():
    lat = Lattice(11)
    hub = Hubbard(lat)
    for _ in range(20):
        op = rand_interior_op(lat, 4, 6)
        for a in ((1, 0), (0, 1), (-2, 1), (2, 2)):
            assert lat.shift_operator(hub.L(op), *a) == \
                hub.L(lat.shift_operator(op, *a)), "translation covariance fail"


def test_L_preserves_grades():
    lat = Lattice(11)
    hub = Hubbard(lat)
    for _ in range(20):
        # random single-grade operator: fix a (dag, c) shape per trial
        modes_up = [lat.mode(x, y, 0) for x in range(4, 7) for y in range(4, 7)]
        modes_dn = [lat.mode(x, y, 1) for x in range(4, 7) for y in range(4, 7)]
        dag = (1 << random.choice(modes_up)) | (1 << random.choice(modes_dn))
        c = 1 << random.choice(modes_up)
        op = Operator({(dag, c): 1})
        g = lat.operator_charges(op)
        out = hub.L(op)
        assert lat.operator_charges(out) == g
        q = lat.particle_charge(next(iter(op.terms)))
        tsz = lat.twice_sz(next(iter(op.terms)))
        for m in out.terms:
            assert lat.particle_charge(m) == q and lat.twice_sz(m) == tsz


def test_L_support_growth_at_most_one():
    lat = Lattice(13)
    hub = Hubbard(lat)
    op = rand_interior_op(lat, 5, 7, nterms=2, deg=3)
    for _ in range(3):
        b0 = lat.bbox(op.support())
        op = hub.L(op)
        if not op:
            break
        b1 = lat.bbox(op.support())
        assert (b0[0] - b1[0] <= 1 and b0[1] - b1[1] <= 1
                and b1[2] - b0[2] <= 1 and b1[3] - b0[3] <= 1), \
            "support grew by more than one lattice step"


def test_L_locality_vs_full_window():
    """Per-monomial locality filter == commutator with fully materialized H."""
    lat = Lattice(7)
    hub = Hubbard(lat)
    Hfull = Operator.zero()
    for x in range(7):
        for y in range(7):
            Hfull = Hfull + hub.interaction(x, y)
            if x + 1 < 7:
                Hfull = Hfull + hub.bond(x, y, "x")
            if y + 1 < 7:
                Hfull = Hfull + hub.bond(x, y, "y")
    assert Hfull.is_hermitian()
    for _ in range(25):
        op = rand_interior_op(lat, 2, 4)
        assert hub.L(op) == commutator(Hfull, op), "locality filter mismatch"


# ----------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------

def test_direction_validation():
    lat = Lattice(5)
    hub = Hubbard(lat)
    try:
        hub.bond(1, 1, "z")
        assert False, "invalid direction accepted"
    except ValueError as e:
        assert "direction" in str(e)


def test_boundary_guards():
    lat = Lattice(5)
    hub = Hubbard(lat)
    edge_op = Operator({(1 << lat.mode(0, 2, 0), 0): 1})
    for fn in (hub.L, hub.neighborhood_H):
        try:
            fn(edge_op)
            assert False, f"{fn.__name__} accepted edge support"
        except ValueError as e:
            assert "window edge" in str(e)


def test_translate_out_of_window_raises():
    lat = Lattice(5)
    m = (1 << lat.mode(3, 3, 0), 0)
    try:
        lat.translate_mono(m, 2, 0)
        assert False, "out-of-window translation accepted"
    except ValueError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"ok  {f.__name__}")
    print(f"\nall {len(fns)} tests passed")
