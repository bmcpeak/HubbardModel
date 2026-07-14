#!/usr/bin/env python3
"""
Independent exact diagonalization oracle for the spinful 2x2 Hubbard model.

Default:
    U=8, t=1, N=3 electrons (Hubbard density n=N/sites=3/4),
    periodic 2x2 torus.

Notation:
    d[x,y,s] = creation operator
    c[x,y,s] = annihilation operator
Coordinates are 1-based; s=0 is down and s=1 is up.
Products use **, for example d[1,1,0]**c[1,2,0].

This program intentionally does not import the bootstrap algebra.py.  It uses
an independent occupation-bitstring implementation of the CAR signs.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

L = 2
N_SITES = 4
N_MODES = 8
DOWN, UP = 0, 1


@dataclass(frozen=True, order=True)
class Factor:
    kind: str  # 'd' or 'c'
    mode: int


Monomial = Tuple[Factor, ...]


def mode_of(x: int, y: int, spin: int) -> int:
    if x not in (1, 2) or y not in (1, 2):
        raise ValueError(f"coordinates must be 1 or 2, got ({x},{y})")
    if spin not in (0, 1):
        raise ValueError("spin must be 0 (down) or 1 (up)")
    return 2 * (L * (x - 1) + (y - 1)) + spin


def coordinates_of(mode: int) -> Tuple[int, int, int]:
    spin = mode & 1
    site = mode >> 1
    return site // L + 1, site % L + 1, spin


def show_factor(f: Factor) -> str:
    x, y, s = coordinates_of(f.mode)
    return f"{f.kind}[{x},{y},{s}]"


class Operator:
    """Sparse sum of ordered elementary-operator products."""

    def __init__(self, terms: Optional[Mapping[Monomial, complex]] = None):
        out: Dict[Monomial, complex] = {}
        if terms:
            for mono, coeff in terms.items():
                out[tuple(mono)] = out.get(tuple(mono), 0j) + complex(coeff)
        self.terms = {m: z for m, z in out.items() if abs(z) > 1e-15}

    @staticmethod
    def identity() -> "Operator":
        return Operator({(): 1})

    @staticmethod
    def scalar(value: complex) -> "Operator":
        return Operator({(): value})

    @staticmethod
    def elementary(kind: str, mode: int) -> "Operator":
        return Operator({(Factor(kind, mode),): 1})

    def __add__(self, other: object) -> "Operator":
        rhs = as_operator(other)
        out = dict(self.terms)
        for mono, coeff in rhs.terms.items():
            out[mono] = out.get(mono, 0j) + coeff
        return Operator(out)

    __radd__ = __add__

    def __neg__(self) -> "Operator":
        return Operator({m: -z for m, z in self.terms.items()})

    def __sub__(self, other: object) -> "Operator":
        return self + (-as_operator(other))

    def __rsub__(self, other: object) -> "Operator":
        return as_operator(other) - self

    def __mul__(self, other: object) -> "Operator":
        if isinstance(other, (int, float, complex, np.number)):
            return Operator({m: z * complex(other) for m, z in self.terms.items()})
        rhs = as_operator(other)
        out: Dict[Monomial, complex] = {}
        for left, a in self.terms.items():
            for right, b in rhs.terms.items():
                mono = left + right
                out[mono] = out.get(mono, 0j) + a * b
        return Operator(out)

    def __rmul__(self, other: object) -> "Operator":
        if isinstance(other, (int, float, complex, np.number)):
            return self * other
        return as_operator(other) * self

    def dagger(self) -> "Operator":
        out: Dict[Monomial, complex] = {}
        for mono, coeff in self.terms.items():
            dag = tuple(
                Factor("d" if f.kind == "c" else "c", f.mode)
                for f in reversed(mono)
            )
            out[dag] = out.get(dag, 0j) + coeff.conjugate()
        return Operator(out)

    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        pieces = []
        for mono, coeff in self.terms.items():
            body = "**".join(show_factor(f) for f in mono) if mono else "I"
            pieces.append(f"({format_number(coeff)})*{body}")
        return " + ".join(pieces)


def as_operator(value: object) -> Operator:
    if isinstance(value, Operator):
        return value
    if isinstance(value, (int, float, complex, np.number)):
        return Operator.scalar(complex(value))
    raise TypeError(f"expected operator or scalar, got {type(value).__name__}")


class ExpressionParser:
    """Safe parser for d[x,y,s], c[x,y,s], I, +, -, *, /, and **."""

    def parse(self, text: str) -> Operator:
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"invalid expression: {exc}") from exc
        return as_operator(self._eval(tree.body))

    def _eval(self, node: ast.AST) -> object:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, complex)):
                return complex(node.value)
            raise ValueError("only numeric constants are allowed")

        if isinstance(node, ast.Name):
            if node.id == "I":
                return Operator.identity()
            if node.id == "j":
                return 1j
            raise ValueError(f"unknown name {node.id!r}")

        if isinstance(node, ast.Subscript):
            if not isinstance(node.value, ast.Name) or node.value.id not in ("d", "c"):
                raise ValueError("only d[x,y,s] and c[x,y,s] are allowed")
            indices = self._indices(node.slice)
            if len(indices) != 3:
                raise ValueError("d and c need exactly three indices")
            x, y, spin = indices
            return Operator.elementary(node.value.id, mode_of(x, y, spin))

        if isinstance(node, ast.UnaryOp):
            value = self._eval(node.operand)
            if isinstance(node.op, ast.USub):
                return -as_operator(value) if isinstance(value, Operator) else -value
            if isinstance(node.op, ast.UAdd):
                return value
            raise ValueError("unsupported unary operation")

        if isinstance(node, ast.BinOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            if isinstance(node.op, ast.Add):
                return as_operator(left) + as_operator(right)
            if isinstance(node.op, ast.Sub):
                return as_operator(left) - as_operator(right)
            if isinstance(node.op, ast.Mult):
                if isinstance(left, Operator) or isinstance(right, Operator):
                    return as_operator(left) * as_operator(right)
                return complex(left) * complex(right)
            if isinstance(node.op, ast.Div):
                if isinstance(right, Operator):
                    raise ValueError("division by an operator is not allowed")
                if isinstance(left, Operator):
                    return left * (1 / complex(right))
                return complex(left) / complex(right)
            if isinstance(node.op, ast.Pow):
                if not isinstance(left, Operator) or not isinstance(right, Operator):
                    raise ValueError("use ** only between operators")
                return left * right
            raise ValueError("unsupported binary operation")

        raise ValueError(f"unsupported syntax: {ast.dump(node, include_attributes=False)}")

    def _indices(self, node: ast.AST) -> Tuple[int, ...]:
        parts = node.elts if isinstance(node, ast.Tuple) else [node]
        values: List[int] = []
        for part in parts:
            value = self._eval(part)
            if isinstance(value, Operator):
                raise ValueError("indices must be integers")
            z = complex(value)
            if abs(z.imag) > 0 or int(z.real) != z.real:
                raise ValueError("indices must be real integers")
            values.append(int(z.real))
        return tuple(values)


def apply_factor(state: int, factor: Factor) -> Optional[Tuple[int, int]]:
    bit = 1 << factor.mode
    occupied = bool(state & bit)
    sign = -1 if (state & (bit - 1)).bit_count() & 1 else 1

    if factor.kind == "c":
        if not occupied:
            return None
        return state ^ bit, sign
    if factor.kind == "d":
        if occupied:
            return None
        return state | bit, sign
    raise ValueError(f"bad factor kind {factor.kind!r}")


def apply_monomial(state: int, monomial: Monomial) -> Optional[Tuple[int, int]]:
    current = state
    total_sign = 1
    for factor in reversed(monomial):
        result = apply_factor(current, factor)
        if result is None:
            return None
        current, sign = result
        total_sign *= sign
    return current, total_sign


class Hubbard2x2Exact:
    def __init__(
        self,
        U: float = 8.0,
        t: float = 1.0,
        N: int = 3,
        bond_convention: str = "torus",
        degeneracy_tolerance: float = 1e-10,
        average_ground_space: bool = True,
    ):
        if not 0 <= N <= N_MODES:
            raise ValueError(f"N must be between 0 and {N_MODES}")
        if bond_convention not in ("torus", "simple_graph"):
            raise ValueError("bond_convention must be 'torus' or 'simple_graph'")

        self.U = float(U)
        self.t = float(t)
        self.N = int(N)
        self.bond_convention = bond_convention
        self.degeneracy_tolerance = float(degeneracy_tolerance)
        self.average_ground_space = bool(average_ground_space)
        self.parser = ExpressionParser()

        self.basis = [s for s in range(1 << N_MODES) if s.bit_count() == self.N]
        self.index = {s: i for i, s in enumerate(self.basis)}
        self.H = self._build_hamiltonian()

        herm_error = np.max(np.abs(self.H - self.H.conj().T))
        if herm_error > 1e-12:
            raise RuntimeError(f"H is not Hermitian; max error {herm_error}")

        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.H)
        self.ground_energy = float(self.eigenvalues[0])
        self.ground_indices = np.flatnonzero(
            np.abs(self.eigenvalues - self.ground_energy) <= self.degeneracy_tolerance
        )

    @staticmethod
    def _neighbor(x: int, y: int, dx: int, dy: int) -> Tuple[int, int]:
        return ((x - 1 + dx) % L + 1, (y - 1 + dy) % L + 1)

    def _bonds(self):
        bonds = []
        for x in (1, 2):
            for y in (1, 2):
                for dx, dy in ((1, 0), (0, 1)):
                    bonds.append(((x, y), self._neighbor(x, y, dx, dy)))
        if self.bond_convention == "torus":
            return bonds
        unique = {}
        for a, b in bonds:
            unique[tuple(sorted((a, b)))] = (a, b)
        return list(unique.values())

    def _add_monomial(self, matrix: np.ndarray, monomial: Monomial, coeff: complex):
        for col, ket in enumerate(self.basis):
            result = apply_monomial(ket, monomial)
            if result is None:
                continue
            state, sign = result
            row = self.index.get(state)
            if row is not None:
                matrix[row, col] += coeff * sign

    def _build_hamiltonian(self) -> np.ndarray:
        H = np.zeros((len(self.basis), len(self.basis)), dtype=np.complex128)

        for row, state in enumerate(self.basis):
            doublons = 0
            for x in (1, 2):
                for y in (1, 2):
                    if (state >> mode_of(x, y, UP)) & 1 and (state >> mode_of(x, y, DOWN)) & 1:
                        doublons += 1
            H[row, row] += self.U * doublons

        for (x1, y1), (x2, y2) in self._bonds():
            for spin in (DOWN, UP):
                m1, m2 = mode_of(x1, y1, spin), mode_of(x2, y2, spin)
                self._add_monomial(H, (Factor("d", m1), Factor("c", m2)), -self.t)
                self._add_monomial(H, (Factor("d", m2), Factor("c", m1)), -self.t)
        return H

    def operator_matrix(self, expression_or_operator: object) -> np.ndarray:
        op = self.parser.parse(expression_or_operator) if isinstance(expression_or_operator, str) else as_operator(expression_or_operator)
        matrix = np.zeros_like(self.H)
        for mono, coeff in op.terms.items():
            self._add_monomial(matrix, mono, coeff)
        return matrix

    def expect(self, expression_or_operator: object) -> complex:
        O = self.operator_matrix(expression_or_operator)
        indices = self.ground_indices if self.average_ground_space else self.ground_indices[:1]
        vals = []
        for i in indices:
            psi = self.eigenvectors[:, i]
            vals.append(np.vdot(psi, O @ psi))
        return complex(np.mean(vals))

    def compare(self, left: str, right: str):
        a, b = self.expect(left), self.expect(right)
        return a, b, a - b

    def dagger(self, expression: str) -> Operator:
        return self.parser.parse(expression).dagger()

    def info(self) -> str:
        return (
            f"2x2 Hubbard: U={self.U:g}, t={self.t:g}, N={self.N}, density={self.N / N_SITES:g}\n"
            f"bond convention: {self.bond_convention}\n"
            f"fixed-N Hilbert dimension: {len(self.basis)}\n"
            f"ground energy: {self.ground_energy:.15g}\n"
            f"ground-space degeneracy (tol={self.degeneracy_tolerance:g}): {len(self.ground_indices)}\n"
            f"state used for expectations: "
            f"{'equal ground-space mixture' if self.average_ground_space else 'first numerical ground eigenvector'}"
        )

    def spectrum(self, count: int = 12) -> str:
        return "\n".join(
            f"{i:3d}: {self.eigenvalues[i]: .15g}"
            for i in range(min(count, len(self.eigenvalues)))
        )


def format_number(value: complex, tol: float = 5e-13) -> str:
    z = complex(value)
    re = 0.0 if abs(z.real) < tol else z.real
    im = 0.0 if abs(z.imag) < tol else z.imag
    if im == 0:
        return f"{re:.15g}"
    if re == 0:
        return f"{im:.15g}j"
    return f"{re:.15g}{'+' if im >= 0 else '-'}{abs(im):.15g}j"


HELP = """
Commands:
  expect EXPR
  dagger EXPR
  compare EXPR1 ; EXPR2
  spectrum
  info
  help
  quit

A bare expression is shorthand for `expect EXPR`.

Examples:
  expect d[1,1,0]**c[1,1,1]
  expect d[1,1,0]**c[1,2,0]
  compare d[1,1,0]**c[1,2,0] ; d[1,2,0]**c[1,1,0]
  expect (d[1,1,0]**c[1,2,0] + d[1,2,0]**c[1,1,0])/2
""".strip()


def repl(solver: Hubbard2x2Exact):
    print(solver.info())
    print("\n" + HELP)
    while True:
        try:
            line = input("\nexact> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        command, _, payload = line.partition(" ")
        command = command.lower()
        try:
            if command in ("quit", "exit", "q"):
                return
            if command == "help":
                print(HELP)
            elif command == "info":
                print(solver.info())
            elif command == "spectrum":
                print(solver.spectrum())
            elif command == "expect":
                if not payload:
                    raise ValueError("usage: expect EXPR")
                print(format_number(solver.expect(payload)))
            elif command == "dagger":
                if not payload:
                    raise ValueError("usage: dagger EXPR")
                print(solver.dagger(payload))
            elif command == "compare":
                if ";" not in payload:
                    raise ValueError("usage: compare EXPR1 ; EXPR2")
                left, right = payload.split(";", 1)
                a, b, diff = solver.compare(left.strip(), right.strip())
                print(f"left:       {format_number(a)}")
                print(f"right:      {format_number(b)}")
                print(f"difference: {format_number(diff)}")
            else:
                print(format_number(solver.expect(line)))
        except Exception as exc:
            print(f"error: {exc}")



def run_self_tests(solver: Hubbard2x2Exact) -> None:
    """Independent sanity checks of CAR, Hermiticity, and basic invariance."""
    dim = len(solver.basis)
    identity = np.eye(dim, dtype=np.complex128)

    same = "c[1,1,0]**d[1,1,0] + d[1,1,0]**c[1,1,0]"
    distinct = "c[1,1,0]**d[1,2,0] + d[1,2,0]**c[1,1,0]"
    err_same = float(np.max(np.abs(solver.operator_matrix(same) - identity)))
    err_distinct = float(np.max(np.abs(solver.operator_matrix(distinct))))
    err_H = float(np.max(np.abs(solver.H - solver.H.conj().T)))

    hop = "d[1,1,0]**c[1,2,0]"
    hop_dag = "d[1,2,0]**c[1,1,0]"
    dagger_error = abs(solver.expect(hop_dag) - solver.expect(hop).conjugate())

    densities = [
        solver.expect(f"d[{x},{y},{spin}]**c[{x},{y},{spin}]")
        for x in (1, 2) for y in (1, 2) for spin in (0, 1)
    ]
    density_spread = float(max(abs(z - np.mean(densities)) for z in densities))

    tests = {
        "Hamiltonian Hermiticity": err_H,
        "same-mode CAR": err_same,
        "distinct-mode CAR": err_distinct,
        "dagger expectation relation": float(dagger_error),
        "site/spin density invariance": density_spread,
    }
    print(solver.info())
    print("\nSelf-tests:")
    failed = False
    for name, error in tests.items():
        ok = error < 1e-10
        failed |= not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: error={error:.3e}")
    if failed:
        raise SystemExit(1)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--U", type=float, default=8.0)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--N", type=int, default=3, help="N=3 is Hubbard density n=3/4 on four sites")
    ap.add_argument("--bond-convention", choices=("torus", "simple_graph"), default="torus")
    ap.add_argument("--degeneracy-tolerance", type=float, default=1e-10)
    ap.add_argument("--pure-ground-state", action="store_true")
    ap.add_argument("--expr")
    ap.add_argument("--spectrum", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    solver = Hubbard2x2Exact(
        U=args.U,
        t=args.t,
        N=args.N,
        bond_convention=args.bond_convention,
        degeneracy_tolerance=args.degeneracy_tolerance,
        average_ground_space=not args.pure_ground_state,
    )

    if args.self_test:
        run_self_tests(solver)
    elif args.expr is not None:
        print(solver.info())
        print(f"\n<{args.expr}> = {format_number(solver.expect(args.expr))}")
    elif args.spectrum:
        print(solver.info())
        print("\nLowest eigenvalues:\n" + solver.spectrum())
    else:
        repl(solver)


if __name__ == "__main__":
    main()
