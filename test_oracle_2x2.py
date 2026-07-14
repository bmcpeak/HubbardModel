"""Cross-check against the independent ED oracle (exact_hubbard_2x2.py).

The oracle (written by ChatGPT, independent bitstring/CAR implementation)
computes ground-space expectation values on the 2x2 PBC torus at
U=8, t=1, N=3, with the L=2 doubled-bond convention, averaging over the
4-fold degenerate ground space (which preserves translations, D4, spin).

This test rebuilds the SAME system entirely from our stack — algebra.py
operators, the Fock applier from test_symmetry — and checks against the
oracle's published reference numbers:

    ground energy            -5.50431591315406
    ground-space degeneracy  4
    <d_s(site1) c_s(site2)>   0.214576442892941   (nearest neighbors, either spin)
    <d_up(x) c_dn(x)>         0                   (on-site spin-off-diagonal)

Convention notes for manual comparison against the oracle:
  * OUR spin convention: s=0 is UP. The oracle uses s=0 = down. All
    reference values above are spin-symmetric, so this is moot here; for
    spin-resolved comparisons map our s <-> 1 - s_oracle.
  * Sites here are (x, y) in {0,1}^2; oracle sites are 1-based. All four
    sites and all NN pairs are equivalent on the torus, so labels don't
    matter for the values checked.

Also runs a product-consistency check that exercises mono_mul against the
state representation: <A*B> computed via our operator product equals
sequential application. Extend the printed table freely and diff against
`python exact_hubbard_2x2.py --expr ...` output.

Run: python tests/test_oracle_2x2.py
"""

import itertools
import random
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from algebra import Operator, dop, cop, nop
from modes import Lattice
from test_symmetry import apply_mono_state

random.seed(1)

E0_ORACLE = -5.50431591315406
DEGEN_ORACLE = 4
HOP_ORACLE = 0.214576442892941


class Torus2x2:
    """2x2 PBC Hubbard at U=8, t=1, N=3; equal mixture over the ground space."""

    def __init__(self, t=1, U=8, N=3):
        lat = Lattice(2)
        self.lat = lat
        # doubled bonds: on L=2 each NN pair is connected twice around the torus
        pairs = [((0, 0), (1, 0)), ((0, 1), (1, 1)),   # x-direction
                 ((0, 0), (0, 1)), ((1, 0), (1, 1))]   # y-direction
        H = Operator.zero()
        for (a, b) in pairs:
            for s in (0, 1):
                hop = dop(lat.mode(*a, s)) * cop(lat.mode(*b, s))
                H = H + (-2 * t) * (hop + hop.dag())   # multiplicity 2
        for x in (0, 1):
            for y in (0, 1):
                H = H + U * (nop(lat.mode(x, y, 0)) * nop(lat.mode(x, y, 1)))
        assert H.is_hermitian()
        self.H = H

        # N-particle sector over 8 modes
        self.states = [f for f in range(256) if bin(f).count("1") == N]
        self.index = {f: k for k, f in enumerate(self.states)}
        dim = len(self.states)
        Hm = np.zeros((dim, dim))
        for k, f in enumerate(self.states):
            for m, v in H.terms.items():
                r = apply_mono_state(m, f)
                if r is not None:
                    Hm[self.index[r[1]], k] += float(v) * r[0]
        assert np.max(np.abs(Hm - Hm.T)) < 1e-12
        w, V = np.linalg.eigh(Hm)
        self.e0 = w[0]
        self.degeneracy = int(np.sum(w < w[0] + 1e-9))
        self.ground = V[:, :self.degeneracy]

    def expval(self, op: Operator) -> float:
        """(1/g) sum_k <psi_k| op |psi_k> over the ground multiplet."""
        total = 0.0
        for col in range(self.degeneracy):
            psi = self.ground[:, col]
            for k, f in enumerate(self.states):
                a = psi[k]
                if abs(a) < 1e-15:
                    continue
                for m, v in op.terms.items():
                    r = apply_mono_state(m, f)
                    if r is not None and r[1] in self.index:
                        total += psi[self.index[r[1]]] * float(v) * r[0] * a
        return total / self.degeneracy


def test_oracle_reference_numbers():
    tor = Torus2x2()
    lat = tor.lat
    assert abs(tor.e0 - E0_ORACLE) < 1e-9, \
        f"ground energy {tor.e0!r} != oracle {E0_ORACLE!r}"
    assert tor.degeneracy == DEGEN_ORACLE, \
        f"degeneracy {tor.degeneracy} != oracle {DEGEN_ORACLE}"
    # NN hopping correlator, both spins, two inequivalent-looking pairs
    for s in (0, 1):
        for (a, b) in (((0, 0), (1, 0)), ((0, 0), (0, 1))):
            v = tor.expval(Operator({(1 << lat.mode(*a, s),
                                      1 << lat.mode(*b, s)): 1}))
            assert abs(v - HOP_ORACLE) < 1e-9, \
                f"<d c> spin {s} pair {a}-{b}: {v!r} != {HOP_ORACLE!r}"
    # on-site spin-off-diagonal vanishes
    v = tor.expval(Operator({(1 << lat.mode(0, 0, 0),
                              1 << lat.mode(0, 0, 1)): 1}))
    assert abs(v) < 1e-9
    print(f"    E0 = {tor.e0:.14f}, degeneracy {tor.degeneracy}, "
          f"<dc>_NN = {HOP_ORACLE}")


def test_product_consistency_in_state():
    """<A*B> via our operator product == <A applied after B>, mixed GS.

    Exercises mono_mul against the Fock representation in a nontrivial
    correlated state rather than on basis states alone.
    """
    tor = Torus2x2()
    lat = tor.lat
    modes = list(range(lat.n_modes))
    for _ in range(25):
        A = Operator({(1 << random.choice(modes), 1 << random.choice(modes)):
                      random.choice((-1, 1, 2))})
        B = Operator({(1 << random.choice(modes), 1 << random.choice(modes)):
                      random.choice((-1, 1))})
        lhs = tor.expval(A * B)
        # sequential: <psi| A B |psi> by applying B then A inside expval
        rhs = tor.expval(Operator(dict((A * B).terms)))  # trivially same object
        # the REAL sequential check: build A*B by hand-application
        total = 0.0
        for col in range(tor.degeneracy):
            psi = tor.ground[:, col]
            # B|psi>
            img = {}
            for k, f in enumerate(tor.states):
                if abs(psi[k]) < 1e-15:
                    continue
                for m, v in B.terms.items():
                    r = apply_mono_state(m, f)
                    if r is not None:
                        img[r[1]] = img.get(r[1], 0.0) + float(v) * r[0] * psi[k]
            # A applied to that
            img2 = {}
            for f, a in img.items():
                for m, v in A.terms.items():
                    r = apply_mono_state(m, f)
                    if r is not None:
                        img2[r[1]] = img2.get(r[1], 0.0) + float(v) * r[0] * a
            total += sum(psi[tor.index[f]] * a for f, a in img2.items()
                         if f in tor.index)
        seq = total / tor.degeneracy
        assert abs(lhs - seq) < 1e-9, "operator product disagrees with sequential application"
        assert abs(lhs - rhs) < 1e-15


def print_comparison_table():
    """Labeled expectation values for manual diff against the oracle CLI.

    Spin map for the oracle: OUR s=0 (up) <-> oracle s=1; sites 0-based here.
    """
    tor = Torus2x2()
    lat = tor.lat
    rows = []
    for s in (0, 1):
        rows.append((f"d[(0,0),s={s}] c[(1,0),s={s}]",
                     Operator({(1 << lat.mode(0, 0, s), 1 << lat.mode(1, 0, s)): 1})))
    rows.append(("n_up(0,0)", nop(lat.mode(0, 0, 0))))
    rows.append(("n_up(0,0) n_dn(0,0)",
                 nop(lat.mode(0, 0, 0)) * nop(lat.mode(0, 0, 1))))
    rows.append(("n_up(0,0) n_up(1,0)",
                 nop(lat.mode(0, 0, 0)) * nop(lat.mode(1, 0, 0))))
    rows.append(("S+S-(0,0)->(1,0): d_up(0,0) c_dn(0,0) d_dn(1,0) c_up(1,0)",
                 Operator({(1 << lat.mode(0, 0, 0), 1 << lat.mode(0, 0, 1)): 1})
                 * Operator({(1 << lat.mode(1, 0, 1), 1 << lat.mode(1, 0, 0)): 1})))
    print("    --- comparison table (diff against oracle; our s=0 is UP) ---")
    for label, op in rows:
        print(f"    {label:58s} {tor.expval(op):+.12f}")


if __name__ == "__main__":
    test_oracle_reference_numbers()
    print("ok  test_oracle_reference_numbers")
    test_product_consistency_in_state()
    print("ok  test_product_consistency_in_state")
    print_comparison_table()
    print("\nall oracle cross-checks passed")
