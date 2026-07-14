"""Patch operator basis: all canonical monomials of degree <= L supported
on an N x N patch of sites.

This is the 'naive' basis for comparison against Krylov bases: variable
count and matrix sizes scale steeply with N and L, but it requires no
choices beyond (N, L). The basis is a list of single-monomial Operators
with integer coefficient 1; grading, mirror folding, and irrep splitting
all happen downstream in symmetry.decompose.

Properties relied on downstream:
  * closed under hermitian conjugation (monomial dagger is a monomial of
    the same degree on the same patch), so both operator orderings
    <O†O> appear among the blocks automatically;
  * closed under the point group about the patch center (the patch is a
    D4-symmetric region), so decompose's close_span is a fast no-op rank
    check rather than a real closure.

The patch center is (x0 + (N-1)/2, y0 + (N-1)/2) — half-integer for even
N, which PointGroup supports.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from algebra import Operator
from modes import Lattice


def patch_modes(lat: Lattice, x0: int, y0: int, N: int):
    if x0 < 0 or y0 < 0 or x0 + N > lat.W or y0 + N > lat.W:
        raise ValueError(f"patch [{x0},{x0+N}) x [{y0},{y0+N}) exceeds window W={lat.W}")
    return [lat.mode(x, y, s)
            for x in range(x0, x0 + N)
            for y in range(y0, y0 + N)
            for s in (0, 1)]


def patch_center(x0: int, y0: int, N: int):
    return (Fraction(2 * x0 + N - 1, 2), Fraction(2 * y0 + N - 1, 2))


def patch_basis(lat: Lattice, x0: int, y0: int, N: int, L: int,
                include_identity: bool = True):
    """All canonical monomials (dag, c) with dag/c modes on the patch and
    total degree 1..L (plus identity if requested), as Operators."""
    modes = patch_modes(lat, x0, y0, N)
    ops = []
    if include_identity:
        ops.append(Operator.identity())
    for deg in range(1, L + 1):
        for p in range(deg + 1):           # p daggers, deg - p annihilators
            q = deg - p
            for dags in combinations(modes, p):
                dag_mask = sum(1 << i for i in dags)
                for cs in combinations(modes, q):
                    c_mask = sum(1 << j for j in cs)
                    ops.append(Operator({(dag_mask, c_mask): 1}))
    return ops
