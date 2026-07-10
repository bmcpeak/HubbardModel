"""
run.py -- basis configuration, streaming triplet fill, assembly, drivers.
The volatile layer: algebra/physics/symmetry are certified by
test_oracle.py and should not need touching.

Memory design (P=4 capable): the streaming path never materializes form
matrices; triplets accumulate in array.array('q') (8 B/element); each
basis is filled, split, and freed before the next (peak = max over bases,
not sum). hcons/wards are generated after the loop from the accumulated
variable set (they need only the set, not the triplets).

Acceptance: P=3 level-2 unsplit -1.120491; P=3 level-3 split -0.938438;
those two numbers gate any new physics from this file.
"""

import sys
import time
import json
from array import array
from itertools import combinations
from fractions import Fraction

from algebra import (multiply, dagger, op_of, pack_sites, to_linear_form,
                     n_charge, sz_charge, show, IDENTITY, W)
from physics import (objective, filling_constraint, linear_H_constraints,
                     ward_constraints, prune_charged, scrub_forms, OX)
import symmetry as SY

def log(msg):
    print(msg, flush=True)

# ============================================================
# Configuration
# ============================================================

P = 3

def set_patch(p):
    global P
    P = p
    assert (P - 1) + OX + 1 < W, "window too small for this patch"

def monomial_keys(nd, nc):
    pool = [(i, j, s) for i in range(P) for j in range(P) for s in (0, 1)]
    return [pack_sites(list(D), list(C))
            for D in combinations(pool, nd)
            for C in combinations(pool, nc)]

def build_bases(level):
    """-> [(keys, has_identity)] : one basis per N-charge sector."""
    assert level in (2, 3)
    bases = [(monomial_keys(1, 1), True),
             (monomial_keys(0, 1), False), (monomial_keys(1, 0), False),
             (monomial_keys(0, 2), False), (monomial_keys(2, 0), False)]
    if level >= 3:
        bases[1] = (bases[1][0] + monomial_keys(1, 2), False)
        bases[2] = (bases[2][0] + monomial_keys(2, 1), False)
        bases += [(monomial_keys(0, 3), False), (monomial_keys(3, 0), False)]
    return bases

# ============================================================
# Fills
# ============================================================

def moment_matrix_keys(keys, has_identity, patch_P, label=None):
    """Form-matrix fill (unsplit path only). Orbit propagation:
    one representative entry per (group x transpose)-orbit."""
    off = 1 if has_identity else 0
    ops = ([IDENTITY] if has_identity else []) + [op_of(k) for k in keys]
    dags = [dagger(o) for o in ops]
    n = len(ops)
    m = [[None] * n for _ in range(n)]

    mats = SY.rep_matrices(keys, patch_P)
    group = list(mats.values())

    if has_identity:
        for j in range(n):
            form = to_linear_form(multiply(dags[0], ops[j]))
            m[0][j] = form
            m[j][0] = form

    t0 = time.time()
    filled = 0
    for i in range(off, n):
        for j in range(off, i + 1):
            if m[i][j] is not None:
                continue
            form = to_linear_form(multiply(dags[i], ops[j]))
            filled += 1
            for cm in group:
                gi, si = cm[i - off]
                gj, sj = cm[j - off]
                s = si * sj
                sform = form if s == 1 else {k: -v for k, v in form.items()}
                for a, b in ((off + gi, off + gj), (off + gj, off + gi)):
                    if m[a][b] is None:
                        m[a][b] = sform
        if label and n >= 500 and (i + 1) % 200 == 0:
            log(f"    {label}: row {i+1}/{n}, filled {filled} "
                f"({time.time()-t0:.0f}s)")
    return m

def moment_triplets_keys(keys, has_identity, patch_P, label=None):
    """Streaming fill: no form matrices. Emits triplets into array('q');
    keys ranked on the fly (rank_key: rank -> key, encounter order).
    Sz-charged variables filtered at the source (replaces prune_charged).
    Returns (Is, Js, Rs, Vals, (a0I, a0J, a0V), rank_key, id_row)."""
    off = 1 if has_identity else 0
    ops = [op_of(k) for k in keys]
    dags = [dagger(o) for o in ops]
    n = len(ops)

    mats = SY.rep_matrices(keys, patch_P)
    group = list(mats.values())

    Is, Js, Rs, Vals = array('q'), array('q'), array('q'), array('q')
    a0I, a0J, a0V = array('q'), array('q'), array('q')
    key_rank, rank_key = {}, []

    def emit(gi, gj, form, s):
        for k, v in form.items():
            if k == 0:
                a0I.append(off + gi); a0J.append(off + gj)
                a0V.append(s * v)
            elif sz_charge(k) == 0:
                r = key_rank.get(k)
                if r is None:
                    r = len(rank_key)
                    key_rank[k] = r
                    rank_key.append(k)
                Is.append(off + gi); Js.append(off + gj)
                Rs.append(r); Vals.append(s * v)

    id_row = None
    if has_identity:
        id_row = [to_linear_form(ops[j]) for j in range(n)]
        for j, form in enumerate(id_row):
            for k, v in form.items():
                if k == 0:
                    a0I.append(0); a0J.append(off + j); a0V.append(v)
                    a0I.append(off + j); a0J.append(0); a0V.append(v)
                elif sz_charge(k) == 0:
                    r = key_rank.get(k)
                    if r is None:
                        r = len(rank_key)
                        key_rank[k] = r
                        rank_key.append(k)
                    Is.append(0); Js.append(off + j)
                    Rs.append(r); Vals.append(v)
                    Is.append(off + j); Js.append(0)
                    Rs.append(r); Vals.append(v)
        a0I.append(0); a0J.append(0); a0V.append(1)

    done = bytearray(n * n)
    t0 = time.time()
    computed = 0
    for i in range(n):
        for j in range(i + 1):
            if done[i * n + j]:
                continue
            form = to_linear_form(multiply(dags[i], ops[j]))
            computed += 1
            for cm in group:
                gi, si = cm[i]
                gj, sj = cm[j]
                s = si * sj
                for a, b in ((gi, gj), (gj, gi)):
                    if not done[a * n + b]:
                        done[a * n + b] = 1
                        emit(a, b, form, s)
        if label and n >= 500 and (i + 1) % 200 == 0:
            log(f"    {label}: row {i+1}/{n}, computed {computed} "
                f"({time.time()-t0:.0f}s)")
    return Is, Js, Rs, Vals, (a0I, a0J, a0V), rank_key, id_row

# ============================================================
# Variables (unsplit path)
# ============================================================

def split_variables(mats):
    vs = set()
    for m in mats:
        for row in m:
            for entry in row:
                vs.update(entry.keys())
    vs.discard(0)
    for v in vs:
        assert n_charge(v) == 0, f"N-charged variable: {show(v)}"
    neutral = {v for v in vs if sz_charge(v) == 0}
    return neutral, vs - neutral

def forms_to_blockdata(mats):
    out = []
    for m in mats:
        n = len(m)
        a0 = ([], [], [])
        pervar = {}
        for i in range(n):
            for j in range(i + 1):
                for k, val in m[i][j].items():
                    tgt = a0 if k == 0 else pervar.setdefault(k, ([], [], []))
                    tgt[0].append(i); tgt[1].append(j)
                    tgt[2].append(float(val))
        out.append((n, a0, pervar))
    return out

# ============================================================
# MOSEK (dual conic form) on blockdata
# ============================================================

def assemble_and_solve(blockdata, obj, equalities, var_index, do_max=False,
                       min_too=True, stream_log=False):
    import mosek

    nv = len(var_index)
    dims = [bd[0] for bd in blockdata]
    nE = len(equalities)

    for form, _ in equalities:
        for k in form:
            assert k == 0 or k in var_index, \
                f"equality references non-variable {show(k)}"
    for _, _, pervar in blockdata:
        for k in pervar:
            assert k in var_index, f"block references non-variable {show(k)}"

    f = [0.0] * nv
    fconst = 0.0
    for k, v in obj.items():
        if k == 0:
            fconst = float(v)
        else:
            f[var_index[k]] = float(v)

    def run_one(fvec):
        with mosek.Env() as env, env.Task() as task:
            if stream_log:
                task.set_Stream(mosek.streamtype.log,
                                lambda s: sys.stdout.write(s))
            task.appendcons(nv)
            task.appendvars(nE)
            for e in range(nE):
                task.putvarbound(e, mosek.boundkey.fr, 0.0, 0.0)
            task.appendbarvars(dims)
            for e, (form, rhs) in enumerate(equalities):
                cst = 0.0
                for k, v in form.items():
                    if k == 0:
                        cst = float(v)
                    else:
                        task.putaij(var_index[k], e, float(v))
                task.putcj(e, float(rhs) - cst)
            for i in range(nv):
                task.putconbound(i, mosek.boundkey.fx, fvec[i], fvec[i])
            for b, (n, a0, pervar) in enumerate(blockdata):
                if a0[0]:
                    idx = task.appendsparsesymmat(n, a0[0], a0[1], a0[2])
                    task.putbarcj(b, [idx], [-1.0])
                for k, (ri, ci, vv) in pervar.items():
                    idx = task.appendsparsesymmat(n, ri, ci, vv)
                    task.putbaraij(var_index[k], b, [idx], [1.0])
            task.putdouparam(mosek.dparam.optimizer_max_time, 5400.0)
            task.putobjsense(mosek.objsense.maximize)
            t1 = time.time()
            task.optimize()
            if stream_log:
                sys.stdout.flush()
            solsta = task.getsolsta(mosek.soltype.itr)
            pobj = task.getprimalobj(mosek.soltype.itr)
            log(f"  solve: {time.time()-t1:.1f}s, status {solsta}, "
                f"certificate {pobj:.6f}")
            if solsta != mosek.solsta.optimal:
                log("  WARNING: not OPTIMAL (time cap? treat max as '>=')")
            return pobj

    lower = run_one(f) + fconst if min_too else None
    upper = None
    if do_max:
        upper = -run_one([-fi for fi in f]) + fconst
    return lower, upper

# ============================================================
# Pipelines
# ============================================================

def prepare_problem(level=3, t=1, U=8, check_equivariance=True,
                    label_prefix=""):
    """nu-independent work, streaming path: fused fill+split per basis
    (peak memory = one basis), then hcons/wards from the variable set.
    Returns dict for solve_at_filling."""
    bases = build_bases(level)
    blockdata, blocksizes = [], []
    neutral = set()
    t_all = time.time()
    for i, (keys, has_id) in enumerate(bases):
        trip = moment_triplets_keys(keys, has_id, P,
                                    label=f"{label_prefix}block {i}")
        Is, Js, Rs, Vals, A0, rank_key, id_row = trip
        neutral.update(rank_key)
        for irrep, dim, a0, pv in SY.split_to_blockdata(
                None, keys, P, has_identity=has_id,
                check_equivariance=check_equivariance, log=log,
                triplets=(Is, Js, Rs, Vals, A0, rank_key),
                id_row=id_row):
            blockdata.append((dim, a0, pv))
            blocksizes.append(dim)
        del trip, Is, Js, Rs, Vals, A0, rank_key
    log(f"fill+split (P={P}, level {level}): {time.time()-t_all:.1f}s, "
        f"blocks {blocksizes}")

    span = neutral | {0}
    hcons, h_oos = linear_H_constraints(neutral, span, t, U)
    wards, dropped = ward_constraints(neutral)
    hcons = scrub_forms(hcons)
    wards = scrub_forms(wards)
    log(f"H-linear {len(hcons)} kept ({h_oos} oos); wards {len(wards)}; "
        f"variables {len(neutral)}")

    var_index = {k: i for i, k in enumerate(sorted(neutral))}
    base_eqs = [(w, 0) for w in wards] + [(h, 0) for h in hcons]
    return dict(blockdata=blockdata, var_index=var_index,
                base_eqs=base_eqs, obj=objective(t, U))

def solve_at_filling(prob, nu, do_max=False, min_too=True,
                     stream_log=False):
    fill = filling_constraint(nu)
    eqs = [(fill[0], fill[1])] + prob["base_eqs"]
    return assemble_and_solve(prob["blockdata"], prob["obj"], eqs,
                              prob["var_index"], do_max=do_max,
                              min_too=min_too, stream_log=stream_log)

def solve_bootstrap_unsplit(level=2, t=1, U=8, nu=Fraction(7, 8),
                            stream_log=False):
    """Reference path: form matrices, no irreps. Regression use only."""
    bases = build_bases(level)
    t0 = time.time()
    mats = [moment_matrix_keys(keys, has_id, P, label=f"block {i}")
            for i, (keys, has_id) in enumerate(bases)]
    log(f"M fill (P={P}, level {level}): {time.time()-t0:.1f}s, "
        f"blocks {[len(m) for m in mats]}")
    neutral, charged = split_variables(mats)
    span = neutral | charged | {0}
    hcons, h_oos = linear_H_constraints(neutral, span, t, U)
    wards, dropped = ward_constraints(neutral)
    log(f"H-linear {len(hcons)} kept ({h_oos} oos); wards {len(wards)}")
    prune_charged(mats)
    hcons = scrub_forms(hcons)
    wards = scrub_forms(wards)
    blockdata = forms_to_blockdata(mats)
    obj = objective(t, U)
    fill = filling_constraint(nu)
    var_index = {k: i for i, k in enumerate(sorted(neutral))}
    eqs = ([(fill[0], fill[1])] + [(w, 0) for w in wards]
           + [(h, 0) for h in hcons])
    log(f"variables: {len(var_index)}")
    return assemble_and_solve(blockdata, obj, eqs, var_index,
                              stream_log=stream_log)

# ============================================================
# nu-scan
# ============================================================

def nu_scan(nus, level=3, t=1, U=8, outfile="nu_scan.json"):
    prob = prepare_problem(level=level, t=t, U=U)
    results = {}
    for nu in nus:
        lo, _ = solve_at_filling(prob, nu)
        results[str(nu)] = float(lo)
        with open(outfile, "w") as fh:
            json.dump(results, fh, indent=1)
        log(f"BANKED nu={nu}: {lo:.6f}")
    pts = sorted((Fraction(s), v) for s, v in results.items())
    log("\nnu        bound       second difference")
    for a in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[a-1], pts[a], pts[a+1]
        d2 = (y2 - y1) / float(x2 - x1) - (y1 - y0) / float(x1 - x0)
        log(f"{float(x1):.4f}   {y1:+.6f}   {d2:+.6f}")
    return results

# ============================================================
# Main: P=4 level-3 launch (gate: P=3 acceptance first)
# ============================================================

if __name__ == "__main__":
    results = {}
    def bank(key, val):
        results[key] = val
        with open("p4_run.json", "w") as fh:
            json.dump(results, fh, indent=1)
        log(f"BANKED {key} = {val:.6f}")

    # Gate: P=3 acceptance through the restructured streaming path.
    set_patch(3)
    log("=== GATE: P=3 level=3 streaming (must be -0.938438) ===")
    prob3 = prepare_problem(level=3, check_equivariance=True)
    lo3, _ = solve_at_filling(prob3, Fraction(7, 8))
    bank("P3_L3_gate", lo3)
    assert abs(lo3 - (-0.938438)) < 5e-6, "gate failed: do not launch P=4"
    del prob3

    # Launch: P=4 level 3. Equivariance off (certified at P=3; the check's
    # lexsorts are the largest remaining transient). MOSEK streaming ON.
    set_patch(4)
    log("\n=== P=4 level=3, irrep-split, streaming ===")
    prob4 = prepare_problem(level=3, check_equivariance=False)
    lo4, _ = solve_at_filling(prob4, Fraction(7, 8), stream_log=True)
    bank("P4_L3", lo4)
    log(f"\nP=4 level-3 bound: {lo4:.6f}   (P=3 was -0.938438; "
        f"variational band -0.74 to -0.77)")