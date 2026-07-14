"""
run.py -- Hubbard bootstrap pipeline: gather PSD blocks, quotient the
Ward/linear identities at BUILD TIME, hand MOSEK the small problem.

    minimize  e = <h_site>   (energy per site, thermodynamic limit)
    over moment functionals subject to
      * M0 blocks: <B_i^dag B_j>          >= 0   (graded basis; yours)
      * M1 blocks: <B_i^dag [H, B_j]>     >= 0   (Stieltjes-on-graded;
                                                  the k=1 spectral rung,
                                                  GROUND-STATE-ONLY)
      * Hankel/Stieltjes tower blocks from krylov.py (H0/H1)
      * monomial Wards <[H, O_m]> = 0 (in-span), vanishing rows,
        filling <n> = NU  -- ALL eliminated before the solver.

Pipeline stages:
  1. sources     -- collect (blockdata, variables) per SOURCES flags
  2. harvest     -- equality rows over the union span (constants live
                    on key 0, matching the a0 convention)
  3. quotient    -- sparse Gauss-Jordan; pivots only on monomial keys
                    (abstract ('Y',...) tuples are never eliminated:
                    they are already quotiented by construction)
  4. substitute  -- rewrite every COO block and the objective in the
                    free variables; constants flow into a0
  5. solve       -- MOSEK Task API (bara triplets); report the DUAL
                    objective as the certified bound. cvxpy fallback
                    for small/debug runs.

SPLICE POINTS (marked ADAPT below):
  * graded_channels(): import and call YOUR patch/graded builder; the
    contract is a list of (label, [operator dict, ...]) with each list
    a symmetry channel. Everything downstream is generic.
  * ward_quotient(): float elimination with Markowitz-lite pivoting.
    If you prefer your exact-Fraction eliminator, swap it in here; the
    substitution consumer only needs {pivot: {key: coeff, ..., 0: c}}.
  * local_energy_op()/density_op(): CHECK the bond-counting convention
    against your old objective. The graded-only baseline reproducing
    -0.938438 is the acceptance test for stage 1+2 conventions; do not
    trust any M1/krylov comparison until that reproduces.

Experiment protocol (flip SOURCES):
  A. graded only                -> must reproduce -0.938438
  B. graded + m1                -> THE kill-criterion run
  C. graded + m1 + krylov       -> full layered scheme
"""

import time

from algebra import (add, multiply, dagger, to_linear_form,
                     normalize_form, c_op, d_op, W, UP, DN)
from physics import ad_H_op
import krylov

# ---------------- configuration ----------------
T_HOP = 1
U_INT = 8
NU = 7.0 / 8.0                 # filling <n_up + n_dn> per site

SOURCES = dict(graded=True, m1=True, krylov=True)
K_DEPTH = 3                    # forwarded to krylov.build_krylov_problem
M1_COMM_CAP = 20_000           # skip B_j in M1 if |[H,B_j]| exceeds this
M1_FORM_BUDGET = 60_000        # skip an M1 entry if its form exceeds this
PIVOT_TOL = 1e-9               # relative pivot threshold in elimination
USE_MOSEK = True               # False -> cvxpy (debug scale only)

CX, CY = W // 2, W // 2

# ============================================================
# Physical forms: per-site energy, density
# ============================================================

def local_energy_op(t=T_HOP, U=U_INT):
    """h_site = -t sum_{delta in {+x,+y}} sum_sigma (c^dag_r c_{r+d} +
    h.c.) + U n_up n_dn at r = center. E/N = <h_site> in the TL.
    ADAPT: verify against your old objective (factor-of-2 bond
    conventions are the classic silent killer)."""
    r = (CX, CY)
    h = {}
    for dlt in ((1, 0), (0, 1)):
        rp = (CX + dlt[0], CY + dlt[1])
        for s in (UP, DN):
            h = add(h, multiply(d_op(*r, s), c_op(*rp, s)))
            h = add(h, multiply(d_op(*rp, s), c_op(*r, s)))
    h = {k: -t * v for k, v in h.items()}
    nu_ = multiply(d_op(*r, UP), c_op(*r, UP))
    nd_ = multiply(d_op(*r, DN), c_op(*r, DN))
    return add(h, {k: U * v for k, v in multiply(nu_, nd_).items()})

def density_op():
    r = (CX, CY)
    return add(multiply(d_op(*r, UP), c_op(*r, UP)),
               multiply(d_op(*r, DN), c_op(*r, DN)))

# ============================================================
# Graded source (ADAPT: wire your builder here)
# ============================================================

def graded_channels():
    """Expected contract: [(label, [op, op, ...]), ...] where each op
    is an operator dict and each list is one symmetry channel (so the
    moment matrix over it is a valid standalone PSD block). Returns
    None if the builder is absent, and the pipeline runs without it."""
    try:
        from patch import build_channels          # ADAPT: your module
    except ImportError:
        return None
    return build_channels()                       # ADAPT: your call

# ============================================================
# Block builders over channel op-lists (COO (n, a0, pv) convention)
# ============================================================

def moment_blockdata(ops):
    """M0: <B_i^dag B_j>, lower triangle."""
    n = len(ops)
    dags = [dagger(o) for o in ops]
    a0, pv = ([], [], []), {}
    for i in range(n):
        for j in range(i + 1):
            form = to_linear_form(multiply(dags[i], ops[j]))
            for k, v in form.items():
                tgt = a0 if k == 0 else pv.setdefault(k, ([], [], []))
                tgt[0].append(i); tgt[1].append(j)
                tgt[2].append(float(v))
    return (n, a0, pv)

def m1_blockdata(ops, t=T_HOP, U=U_INT):
    """M1: <B_i^dag [H, B_j]> over the sub-basis whose commutators stay
    within budget. Emits the lower triangle in this one orientation;
    the mirror equals it on-shell (shuffle), so assembly's symmetric
    mirroring is valid -- same argument as the toy's Hermitian-part
    trick. Returns (blockdata, kept_indices) or (None, []) if nothing
    survives the budget."""
    comms, kept = [], []
    for j, o in enumerate(ops):
        c = ad_H_op(o, t, U)
        if c and len(c) <= M1_COMM_CAP:
            comms.append(c)
            kept.append(j)
    if not kept:
        return None, []
    n = len(kept)
    dags = [dagger(ops[j]) for j in kept]
    a0, pv = ([], [], []), {}
    for i in range(n):
        for j in range(i + 1):
            prod = multiply(dags[i], comms[j])
            form = to_linear_form(prod)
            if len(form) > M1_FORM_BUDGET:
                # An entry silently dropped from a PSD block becomes an
                # implicit ZERO -- that is an invalid constraint, not a
                # weaker one. Operator-level filtering (M1_COMM_CAP) is
                # the safe knob: lower it instead.
                raise RuntimeError(
                    f"M1 entry ({i},{j}) form has {len(form)} terms "
                    f"> M1_FORM_BUDGET; lower M1_COMM_CAP")
            for k, v in form.items():
                tgt = a0 if k == 0 else pv.setdefault(k, ([], [], []))
                tgt[0].append(i); tgt[1].append(j)
                tgt[2].append(float(v))
    return (n, a0, pv), kept

# NOTE on the M1 budget: M1_COMM_CAP filters whole operators, which is
# safe (sub-basis => principal submatrix => still PSD in the ground
# state). M1_FORM_BUDGET is a tripwire that aborts rather than emit an
# unsound block.

# ============================================================
# Equality harvest
# ============================================================

def harvest_monomial_wards(span, t=T_HOP, U=U_INT, log=print):
    """<[H, O_m]> = 0 for every monomial key in span, kept when the
    commutator form is in-span. Constants live on key 0."""
    seen, rows, oos = set(), [], 0
    for m in span:
        if isinstance(m, tuple):
            continue                      # abstract Y: no Ward
        form = to_linear_form(ad_H_op({m: 1}, t, U))
        if not form:
            continue
        if not set(form) <= (span | {0}):
            oos += 1
            continue
        h = normalize_form(form)
        if h not in seen:
            seen.add(h)
            rows.append({k: float(v) for k, v in form.items()})
    log(f"    ward harvest: {len(rows)} kept, {oos} out-of-span")
    return rows

def filling_row(span):
    """<n_center> - NU = 0, as a row with the constant on key 0."""
    form = to_linear_form(density_op())
    row = {k: float(v) for k, v in form.items()}
    row[0] = row.get(0, 0.0) - NU
    missing = set(row) - (span | {0})
    return row, missing

# ============================================================
# Ward quotient: sparse Gauss-Jordan over dict rows
# ============================================================

def ward_quotient(rows, protected=frozenset(), log=print):
    """rows: list of {key: coeff} with key 0 = constant. Returns
    submap {pivot: {key: coeff, ..., 0: const}} meaning
        x_pivot = sum coeff * x_key + const.
    Pivots are chosen among non-protected, non-constant keys,
    shortest-row-first (Markowitz-lite), largest |coeff| within a row.
    Float arithmetic with relative tolerance; swap in your exact
    eliminator here if you prefer (ADAPT) -- the consumer only needs
    the submap. Raises on an inconsistent system (row reducing to a
    nonzero constant), which at this stage means a convention bug."""
    rows = sorted((dict(r) for r in rows), key=len)
    piv_row, order = {}, []            # pivot -> reduced row; insertion order

    def reduce_against(r):
        # forward reduction against existing pivots, iterated to closure
        changed = True
        while changed:
            changed = False
            for p in list(r):
                if p in piv_row and p != 0:
                    f = r.pop(p)
                    for k, v in piv_row[p].items():
                        if k == p:
                            continue
                        r[k] = r.get(k, 0.0) + f * v
                        if abs(r[k]) < 1e-300:
                            del r[k]
                    changed = True
        return r

    for r in rows:
        r = reduce_against(r)
        scale = max((abs(v) for k, v in r.items() if k != 0), default=0.0)
        r = {k: v for k, v in r.items()
             if abs(v) > PIVOT_TOL * max(scale, 1.0)}
        cands = [k for k in r if k != 0 and k not in protected]
        if not cands:
            if 0 in r and scale == 0.0 and abs(r[0]) > 1e-6:
                raise ValueError(f"inconsistent equality, residual {r[0]}")
            continue
        p = max(cands, key=lambda k: abs(r[k]))
        c = r.pop(p)
        # store as x_p = sum (-v/c) x_k  (+ const on key 0)
        piv_row[p] = {p: 1.0, **{k: -v / c for k, v in r.items()}}
        order.append(p)

    # backward pass: eliminate later pivots from earlier rows
    for p in reversed(order):
        row = piv_row[p]
        for q in [k for k in row if k != p and k != 0 and k in piv_row]:
            f = row.pop(q)
            for k, v in piv_row[q].items():
                if k == q:
                    continue
                row[k] = row.get(k, 0.0) + f * v
        piv_row[p] = {k: v for k, v in row.items()
                      if k == p or abs(v) > 1e-14}

    submap = {p: {k: v for k, v in row.items() if k != p}
              for p, row in piv_row.items()}
    log(f"    quotient: {len(submap)} variables eliminated")
    return submap

# ============================================================
# Substitution into COO blocks and forms
# ============================================================

def substitute_form(form, submap):
    out = {}
    for k, v in form.items():
        if k in submap:
            for k2, v2 in submap[k].items():
                out[k2] = out.get(k2, 0.0) + v * v2
        else:
            out[k] = out.get(k, 0.0) + v
    return {k: v for k, v in out.items() if abs(v) > 1e-14 or k == 0}

def substitute_blockdata(bd, submap):
    n, a0, pv = bd
    acc = {}                                  # key -> {(i,j): val}
    def put(key, i, j, v):
        d = acc.setdefault(key, {})
        d[(i, j)] = d.get((i, j), 0.0) + v
    for i, j, v in zip(*a0):
        put(0, i, j, v)
    for key, (ii, jj, vv) in pv.items():
        if key in submap:
            for i, j, v in zip(ii, jj, vv):
                for k2, v2 in submap[key].items():
                    put(k2, i, j, v * v2)
        else:
            for i, j, v in zip(ii, jj, vv):
                put(key, i, j, v)
    a0n, pvn = ([], [], []), {}
    for key, d in acc.items():
        tgt = a0n if key == 0 else pvn.setdefault(key, ([], [], []))
        for (i, j), v in d.items():
            if abs(v) > 1e-14:
                tgt[0].append(i); tgt[1].append(j); tgt[2].append(v)
    return (n, a0n, pvn)

# ============================================================
# Solvers
# ============================================================

def solve_mosek(blocks, obj_form, log=print):
    """min obj s.t. each block M_b(x) >= 0.  x = free vars (union of
    keys in blocks/objective, key 0 excluded). Encoding: barX_b PSD
    slack, one linear constraint per present (b, i>=j) pattern entry:
    <E_ij, barX_b> - sum_v C_v[i,j] x_v = C_0[i,j]. Certified bound =
    DUAL objective at the interior-point solution."""
    import mosek

    keys = set(obj_form) - {0}
    for _, a0, pv in blocks:
        keys |= set(pv)
    keys = sorted(keys, key=repr)
    vidx = {k: i for i, k in enumerate(keys)}
    nv = len(keys)

    with mosek.Env() as env, env.Task(0, 0) as task:
        task.appendvars(nv)
        task.putvarboundsliceconst(0, nv, mosek.boundkey.fr, -0.0, 0.0)
        task.appendbarvars([b[0] for b in blocks])

        ncon = 0
        for bi, (n, a0, pv) in enumerate(blocks):
            pattern = {}
            def _acc(i, j, key, v):
                e = pattern.setdefault((max(i, j), min(i, j)), {})
                e[key] = e.get(key, 0.0) + v
            for i, j, v in zip(*a0):
                _acc(i, j, 0, v)
            for key, (ii, jj, vv) in pv.items():
                for i, j, v in zip(ii, jj, vv):
                    _acc(i, j, key, v)
            for (i, j), ent in sorted(pattern.items()):
                task.appendcons(1)
                w = 1.0 if i == j else 0.5
                sym = task.appendsparsesymmat(n, [i], [j], [w])
                task.putbaraij(ncon, bi, [sym], [1.0])
                cols, vals = [], []
                for key, v in ent.items():
                    if key == 0:
                        continue
                    cols.append(vidx[key]); vals.append(-v)
                task.putarow(ncon, cols, vals)
                c0 = ent.get(0, 0.0)
                task.putconbound(ncon, mosek.boundkey.fx, c0, c0)
                ncon += 1

        for key, v in obj_form.items():
            if key != 0:
                task.putcj(vidx[key], v)
        task.putobjsense(mosek.objsense.minimize)
        log(f"    mosek: {nv} scalar vars, {ncon} constraints, "
            f"{len(blocks)} bar blocks {[b[0] for b in blocks]}")
        task.optimize()
        st = task.getsolsta(mosek.soltype.itr)
        if st not in (mosek.solsta.optimal,):
            log(f"    WARNING: solsta = {st}")
        primal = task.getprimalobj(mosek.soltype.itr)
        dualob = task.getdualobj(mosek.soltype.itr)
        const = obj_form.get(0, 0.0)
        return dualob + const, primal + const, str(st)

def solve_cvxpy(blocks, obj_form, log=print):
    """Debug-scale fallback; same encoding via dense matrices."""
    import numpy as np
    import cvxpy as cp
    keys = set(obj_form) - {0}
    for _, a0, pv in blocks:
        keys |= set(pv)
    keys = sorted(keys, key=repr)
    vidx = {k: i for i, k in enumerate(keys)}
    x = cp.Variable(len(keys))
    cons = []
    for n, a0, pv in blocks:
        C0 = np.zeros((n, n))
        for i, j, v in zip(*a0):
            C0[i, j] += v
            if i != j:
                C0[j, i] += v
        expr = cp.Constant(C0)
        for key, (ii, jj, vv) in pv.items():
            C = np.zeros((n, n))
            for i, j, v in zip(ii, jj, vv):
                C[i, j] += v
                if i != j:
                    C[j, i] += v
            expr = expr + x[vidx[key]] * C
        cons.append((expr + expr.T) / 2 >> 0)
    obj = obj_form.get(0, 0.0)
    for key, v in obj_form.items():
        if key != 0:
            obj = obj + v * x[vidx[key]]
    prob = cp.Problem(cp.Minimize(obj), cons)
    for s in ("MOSEK", "CLARABEL", "SCS"):
        if s not in cp.installed_solvers():
            continue
        try:
            prob.solve(solver=s)
        except Exception:
            continue
        if prob.status in ("optimal", "optimal_inaccurate"):
            return prob.value, prob.value, f"{s}:{prob.status}"
    return float("nan"), float("nan"), "failed"

# ============================================================
# Main
# ============================================================

def main(log=print):
    t0 = time.time()
    blocks, labels = [], []
    variables = set()
    extra_equalities = []

    # ---- stage 1: sources ----
    gch = graded_channels() if (SOURCES["graded"] or SOURCES["m1"]) else None
    if (SOURCES["graded"] or SOURCES["m1"]) and gch is None:
        log("  graded builder absent (see graded_channels ADAPT note); "
            "continuing without graded/M1 sources")
    if gch is not None and SOURCES["graded"]:
        for lab, ops in gch:
            bd = moment_blockdata(ops)
            blocks.append(bd); labels.append((lab, "M0"))
            variables |= set(bd[2])
    if gch is not None and SOURCES["m1"]:
        dropped = 0
        for lab, ops in gch:
            bd, kept = m1_blockdata(ops)
            if bd is None:
                continue
            if len(kept) < len(ops):
                dropped += len(ops) - len(kept)
            blocks.append(bd); labels.append((lab, "M1"))
            variables |= set(bd[2])
        log(f"  M1 source: {dropped} operators over M1_COMM_CAP dropped")
    if SOURCES["krylov"]:
        kbd, kvars, keq, kidx = krylov.build_krylov_problem(
            krylov.seed_star(), K_DEPTH, T_HOP, U_INT, log=log)
        blocks.extend(kbd); labels.extend(kidx)
        variables |= set(kvars)
        extra_equalities.extend(
            {k: float(v) for k, v in f.items()} for f in keq)

    obj_form = {k: float(v)
                for k, v in to_linear_form(local_energy_op()).items()}
    variables |= set(obj_form) - {0}

    # ---- stage 2: harvest ----
    rows = harvest_monomial_wards(variables, log=log)
    rows.extend(extra_equalities)
    frow, missing = filling_row(variables)
    if missing:
        log(f"  WARNING: filling constraint references keys outside the "
            f"span ({len(missing)}); adding them as free variables")
        variables |= missing
    rows.append(frow)

    # ---- stage 3+4: quotient and substitute ----
    protected = {k for k in variables if isinstance(k, tuple)}
    submap = ward_quotient(rows, protected=protected, log=log)
    blocks = [substitute_blockdata(b, submap) for b in blocks]
    obj_form = substitute_form(obj_form, submap)
    free = set(obj_form) - {0}
    for _, a0, pv in blocks:
        free |= set(pv)
    log(f"  after quotient: {len(free)} free variables "
        f"({sum(isinstance(k, tuple) for k in free)} abstract Y), "
        f"{len(blocks)} blocks")

    # ---- stage 5: solve ----
    solver = solve_mosek if USE_MOSEK else solve_cvxpy
    dual, primal, status = solver(blocks, obj_form, log=log)
    log(f"\n  e/site >= {dual:.9f}   (certified, dual objective)")
    log(f"  primal objective  {primal:.9f}   [{status}]")
    log(f"  total {time.time()-t0:.1f}s")
    return dual

if __name__ == "__main__":
    main()
