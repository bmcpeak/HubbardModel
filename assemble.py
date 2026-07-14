"""Assemble the bootstrap SDP from symmetry blocks and the moment table.

Produces a solver-neutral SDPProblem:

    variables      y_k = moment values (k indexes MomentTable ids)
    minimize       obj . y            (energy density <h>)
    subject to     y[identity] = 1
                   filling_row . y = nu
                   M_b(y) >= 0        for each symmetry block b, where
                   M_b(y)[i,j] = resolve(O_i† O_j) . y

Everything upstream is exact (int/Fraction); floats appear only in the
emitted SDPProblem. The moment table is shared across all blocks — that
sharing IS the coupling between blocks, so a single table instance must
be used for the entire problem.

Memory note: this is the in-memory assembler, adequate through the patch
scales a laptop can solve anyway. The streaming two-pass variant (spill
(b, i, j, mid, coeff) triplets to disk, substitute Ward map in pass 2)
slots in behind the same SDPProblem output when profiling demands it.

Validity note: the feasible set is the convex hull of symmetric-state
moment vectors, so at fixed filling the bound is a lower bound on the
CONVEX ENVELOPE of e(nu) (Maxwell construction) — the honest statement
in a phase-separation regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from algebra import Operator
from moments import MomentTable


@dataclass
class BlockData:
    label: str
    charges: tuple
    size: int
    # (i, j) with i <= j  ->  {moment_id: float}
    entries: dict = field(default_factory=dict)


@dataclass
class SDPProblem:
    nvars: int
    blocks: list
    objective: dict                  # {moment_id: float}
    eq_constraints: list             # [({moment_id: float}, rhs)]
    meta: dict = field(default_factory=dict)

    def summary(self):
        bl = ", ".join(f"{b.label}:{b.size}" for b in self.blocks)
        return (f"SDPProblem: {self.nvars} moment variables, "
                f"{len(self.blocks)} blocks [{bl}], "
                f"{len(self.eq_constraints)} equalities")


def _to_float_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, complex):
            raise TypeError(f"complex coefficient {v!r} in a real assembly")
        out[k] = float(v)
    return out


def assemble_block(block, table: MomentTable, progress=False) -> BlockData:
    """Moment-matrix entries of one symmetry block; only i <= j, using
    hermiticity of the state: <O_j† O_i> = <(O_i† O_j)†> = <O_i† O_j>
    under dagger identification (asserted in assemble_problem).

    Fast path (monomial bases): block.seeds records op_i = beta_i * P m_i.
    Since P† P = P for all our projectors (including E row projectors,
    P_11† P_11 = P_11), <op_i† op_j> = beta_i <m_i† op_j> — one single-
    monomial product against op_j instead of |op_i| x |op_j| products.
    Exactly equal to the generic path (regression-tested)."""
    import time
    ops = block.ops
    bd = BlockData(label=block.irrep, charges=block.charges, size=len(ops))
    t0 = time.perf_counter()
    if block.seeds is not None:
        left = [Operator({(m[1], m[0]): 1}) for (m, _) in block.seeds]
        betas = [b for (_, b) in block.seeds]
        for i in range(len(ops)):
            if progress and len(ops) >= 120 and i % 40 == 0 and i:
                print(f"      row {i}/{len(ops)}, {len(bd.entries)} entries, "
                      f"{len(table)} moments ({time.perf_counter()-t0:.0f}s)",
                      flush=True)
            for j in range(i, len(ops)):
                row = table.resolve(left[i] * ops[j])
                if row:
                    bd.entries[(i, j)] = _to_float_row(
                        {k: betas[i] * v for k, v in row.items()})
    else:
        dags = [op.dag() for op in ops]
        for i in range(len(ops)):
            for j in range(i, len(ops)):
                row = table.resolve(dags[i] * ops[j])
                if row:
                    bd.entries[(i, j)] = _to_float_row(row)
    return bd


def assemble_problem(blocks, table: MomentTable, hub, center_site,
                     filling, progress=False) -> SDPProblem:
    """blocks: output of symmetry.decompose; hub: Hubbard instance;
    center_site: (x, y) where the energy density and filling operators are
    anchored (irrelevant up to translation identification, but must have
    forward bonds inside the window); filling: target <n_up + n_dn>."""
    from algebra import nop
    if not table.use_dagger:
        raise ValueError("real symmetric assembly requires the dagger/"
                         "reality identification (MomentTable use_dagger=True)")
    lat = hub.lat
    x0, y0 = center_site

    import time
    block_data = []
    for b in blocks:
        t0 = time.perf_counter()
        bd = assemble_block(b, table, progress=progress)
        if progress:
            print(f"    block {b.charges}/{b.irrep} size {bd.size}: "
                  f"{len(bd.entries)} entries, {len(table)} moments "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
        block_data.append(bd)

    obj = _to_float_row(table.resolve(hub.energy_density(x0, y0)))

    ident = {table.identity_id: 1.0}
    n_op = nop(lat.mode(x0, y0, 0)) + nop(lat.mode(x0, y0, 1))
    fill_row = _to_float_row(table.resolve(n_op))

    return SDPProblem(
        nvars=len(table),
        blocks=block_data,
        objective=obj,
        eq_constraints=[(ident, 1.0), (fill_row, float(filling))],
        meta={"t": hub.t, "U": hub.U, "filling": float(filling)},
    )


def validate_problem(prob: SDPProblem):
    """Structural validation; raises on anything that would produce a
    silently wrong or unbounded problem. Cheap; run it always."""
    import numpy as np
    n = prob.nvars
    covered = set()

    def _check_row(row, where):
        if not row:
            raise ValueError(f"empty coefficient row in {where}")
        for k, c in row.items():
            if not (0 <= k < n):
                raise ValueError(f"moment id {k} out of range in {where}")
            if not np.isfinite(c):
                raise ValueError(f"non-finite coefficient in {where}")
    for b in prob.blocks:
        for (i, j), row in b.entries.items():
            if not (0 <= i <= j < b.size):
                raise ValueError(f"bad entry index ({i},{j}) in block {b.label}")
            _check_row(row, f"block {b.charges}/{b.label} entry ({i},{j})")
            covered.update(row.keys())
    identity_fixes = 0
    for row, rhs in prob.eq_constraints:
        _check_row(row, "equality constraint")
        covered.update(row.keys())
        if set(row.keys()) == {0} and rhs == 1.0:
            identity_fixes += 1
    if identity_fixes != 1:
        raise ValueError(f"identity moment fixed {identity_fixes} times, want 1")
    _check_row(prob.objective, "objective")
    missing = [k for k in prob.objective if k not in covered]
    if missing:
        raise ValueError(
            f"objective moments {missing} appear in NO block entry or "
            f"equality: the problem is unbounded below. (Classic cause: "
            f"basis level too low to control the interaction term, e.g. "
            f"L=1 cannot cover the degree-4 U-term.)")
    return True
