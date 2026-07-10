"""
run.py -- configuration, streaming fill, quotient, assembly, drivers.

Pipeline (irrep path): fill triplets per basis (orbit-propagated, charged
vars filtered at emit) -> D4xZ2 split (symmetry.py) -> ward quotient
applied to per-variable blockdata (spin.py) -> MOSEK. Equalities are
filling + hcons only; wards are gone by substitution.

Gates: P=3 L2 unsplit -1.120491; P=3 L3 split+quotient -0.938438 (exact --
the quotient is an exact transformation).
"""

import sys, time, json
from array import array
from itertools import combinations
from fractions import Fraction

from algebra import (multiply, dagger, op_of, pack_sites, to_linear_form,
                     n_charge, sz_charge, show, IDENTITY, W)
from physics import (objective, filling_constraint, linear_H_constraints,
                     ward_constraints, prune_charged, scrub_forms, OX)
import symmetry as SY
import spin as SP

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

def monomial_keys(nd, nc, max_diam=None):
    pool = [(i, j, s) for i in range(P) for j in range(P) for s in (0, 1)]
    keys = [pack_sites(list(D), list(C))
            for D in combinations(pool, nd)
            for C in combinations(pool, nc)]
    if max_diam is not None:
        def diam(k):
            ss = list({(i, j) for (i, j) in
                       __import__("algebra").support_sites(k)})
            return max((max(abs(a[0]-b[0]), abs(a[1]-b[1]))
                        for a in ss for b in ss), default=0)
        keys = [k for k in keys if diam(k) <= max_diam]
    return keys

def build_bases(level, max_diam=None):
    """-> [(keys, has_identity)]. max_diam restricts degree-3 supports
    (a valid principal-submatrix restriction; no-op at P<=3 for
    max_diam>=2, which is the gate's correctness test)."""
    assert level in (2, 3)
    bases = [(monomial_keys(1, 1), True),
             (monomial_keys(0, 1), False), (monomial_keys(1, 0), False),
             (monomial_keys(0, 2), False), (monomial_keys(2, 0), False)]
    if level >= 3:
        bases[1] = (bases[1][0] + monomial_keys(1, 2, max_diam), False)
        bases[2] = (bases[2][0] + monomial_keys(2, 1, max_diam), False)
        bases += [(monomial_keys(0, 3, max_diam), False),
                  (monomial_keys(3, 0, max_diam), False)]
    return bases

# ============================================================
# Streaming fill (triplets; charged vars filtered at emit)
# ============================================================

def moment_triplets_keys(keys, has_identity, patch_P, mats=None, label=None):
    off = 1 if has_identity else 0
    ops = [op_of(k) for k in keys]
    dags = [dagger(o) for o in ops]
    n = len(ops)
    if mats is None:
        mats = SY.rep_matrices(keys, patch_P)
    group = list(mats.values())

    Is, Js, Rs, Vals = array('q'), array('q'), array('q'), array('q')
    a0I, a0J, a0V = array('q'), array('q'), array('q')
    key_rank, rank_key = {}, []

    def emit(gi, gj, form, s):
        for k, v in form.items():
            if k == 0:
                a0I.append(off + gi); a0J.append(off + gj); a0V.append(s * v)
            elif sz_charge(k) == 0:
                r = key_rank.get(k)
                if r is None:
                    r = len(rank_key); key_rank[k] = r; rank_key.append(k)
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
                        r = len(rank_key); key_rank[k] = r; rank_key.append(k)
                    Is.append(0); Js.append(off + j); Rs.append(r)
                    Vals.append(v)
                    Is.append(off + j); Js.append(0); Rs.append(r)
                    Vals.append(v)
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
                gi, si = cm[i]; gj, sj = cm[j]
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
# Ward quotient application (post-split, on blockdata)
# ============================================================

def substitute_form(form, subst):
    out = {}
    for k, v in form.items():
        if k == 0:
            out[0] = out.get(0, Fraction(0)) + v
            continue
        for k2, w in subst[k].items():
            out[k2] = out.get(k2, Fraction(0)) + v * w
    return {k: v for k, v in out.items() if v != 0}

def apply_quotient_blockdata(blockdata, subst):
    """blockdata: [(dim, a0, {key: (ri, ci, vv)})]. Re-key each variable's
    triplets through subst, merging duplicates on (i, j)."""
    out = []
    for dim, a0, pv in blockdata:
        acc = {}
        for key, (ri, ci, vv) in pv.items():
            for k2, w in subst[key].items():
                fw = float(w)
                d = acc.setdefault(k2, {})
                for r, c, v in zip(ri, ci, vv):
                    d[(r, c)] = d.get((r, c), 0.0) + fw * v
        pv2 = {}
        for k2, d in acc.items():
            items = [(r, c, v) for (r, c), v in d.items() if v != 0.0]
            if items:
                pv2[k2] = ([r for r, c, v in items],
                           [c for r, c, v in items],
                           [v for r, c, v in items])
        out.append((dim, a0, pv2))
    return out

# ============================================================
# MOSEK (unchanged mechanics)
# ============================================================

def assemble_and_solve(blockdata, obj, equalities, var_index, do_max=False,
                       min_too=True, stream_log=False):
    import mosek
    nv = len(var_index)
    dims = [bd[0] for bd in blockdata]
    nE = len(equalities)
    for form, _ in equalities:
        for k in form:
            assert k == 0 or k in var_index, f"non-variable {show(k)}"
    for _, _, pervar in blockdata:
        for k in pervar:
            assert k in var_index, f"block non-variable {show(k)}"
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
            task.appendcons(nv); task.appendvars(nE)
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
            solsta = task.getsolsta(mosek.soltype.itr)
            pobj = task.getprimalobj(mosek.soltype.itr)
            log(f"  solve: {time.time()-t1:.1f}s, status {solsta}, "
                f"certificate {pobj:.6f}")
            if solsta != mosek.solsta.optimal:
                log("  WARNING: not OPTIMAL (capped max = '>=' only)")
            return pobj

    lower = run_one(f) + fconst if min_too else None
    upper = -run_one([-fi for fi in f]) + fconst if do_max else None
    return lower, upper

# ============================================================
# Pipeline
# ============================================================

def prepare_problem(level=3, t=1, U=8, max_diam=None,
                    check_equivariance=True, use_quotient=True, use_spin_blocks=False):
    bases = build_bases(level, max_diam)
    blockdata, blocksizes = [], []
    neutral = set()
    t_all = time.time()
    for i, (keys, has_id) in enumerate(bases):
        mats = SY.rep_matrices(keys, P)
        spin_U = spin_labels = None
        if use_spin_blocks:
            spin_U, spin_labels = SP.spin_adapt_matrix(keys, log=log)
        trip = moment_triplets_keys(keys, has_id, P, mats=mats,
                                    label=f"block {i}")
        Is, Js, Rs, Vals, A0, rank_key, id_row = trip
        neutral.update(rank_key)
        for irrep, dim, a0, pv in SY.split_to_blockdata(
                None, keys, P, has_identity=has_id,
                check_equivariance=check_equivariance, log=log,
                triplets=(Is, Js, Rs, Vals, A0, rank_key),
                id_row=id_row, mats=mats,
                spin_U=spin_U, spin_labels=spin_labels):
            blockdata.append((dim, a0, pv))
            blocksizes.append(dim)
        del trip, Is, Js, Rs, Vals, A0, rank_key
    log(f"fill+split (P={P}, level {level}): {time.time()-t_all:.1f}s, "
        f"largest block {max(blocksizes)}, {len(blocksizes)} blocks")

    span = neutral | {0}
    hcons, h_oos = linear_H_constraints(neutral, span, t, U)
    hcons = scrub_forms(hcons)
    obj = objective(t, U)
    fillf, _ = filling_constraint(Fraction(1, 2))   # form only; rhs later

    if use_quotient:
        wards, _ = ward_constraints(neutral)
        wards = scrub_forms(wards)
        kept, subst = SP.build_ward_quotient(wards, sorted(neutral), log=log)
        blockdata = apply_quotient_blockdata(blockdata, subst)
        obj = substitute_form(obj, subst)
        fillf = substitute_form(fillf, subst)
        hcons = [f2 for h in hcons if (f2 := substitute_form(h, subst))]
        var_index = {k: i for i, k in enumerate(kept)}
    else:
        wards, _ = ward_constraints(neutral)
        wards = scrub_forms(wards)
        hcons = hcons + wards        # wards ride as equalities
        var_index = {k: i for i, k in enumerate(sorted(neutral))}

    log(f"H-linear {len(hcons)} eqs ({h_oos} oos); "
        f"variables {len(var_index)}")
    return dict(blockdata=blockdata, var_index=var_index,
                obj=obj, fill_form=fillf,
                eqs=[(h, 0) for h in hcons])

def solve_at_filling(prob, nu, do_max=False, min_too=True,
                     stream_log=False):
    eqs = [(prob["fill_form"], Fraction(nu))] + prob["eqs"]
    return assemble_and_solve(prob["blockdata"], prob["obj"], eqs,
                              prob["var_index"], do_max=do_max,
                              min_too=min_too, stream_log=stream_log)

# ============================================================
# Main: gates, then P=4
# ============================================================

if __name__ == "__main__":
    results = {}
    def bank(key, val):
        results[key] = val
        with open("p4_spin_run.json", "w") as fh:
            json.dump(results, fh, indent=1)
        log(f"BANKED {key} = {val:.6f}")

    # Gate: certified config at P=3 (cheap insurance before a long run)
    set_patch(3)
    log("=== GATE: P=3 L3 quotient+spin (must be -0.938438) ===")
    prob = prepare_problem(level=3, use_quotient=True,
                           use_spin_blocks=True)
    lo, _ = solve_at_filling(prob, Fraction(7, 8))
    bank("gate", lo)
    assert abs(lo + 0.938438) < 5e-6, "gate failed: do not launch P=4"
    del prob

    set_patch(4)
    log("\n=== P=4 L3, diam<=2, quotient + spin blocks ===")
    prob = prepare_problem(level=3, max_diam=2,
                           check_equivariance=False,
                           use_quotient=True, use_spin_blocks=True)
    lo, _ = solve_at_filling(prob, Fraction(7, 8), stream_log=True)
    bank("P4_L3_diam2_spin", lo)
    log(f"\nP=4: {lo:.6f}  (P=3: -0.938438; variational band: "
        f"-0.74 to -0.77)")


# if __name__ == "__main__":
#     set_patch(3)
#     log("=== nu-scan: P=3 L3, quotient + spin blocks ===")
#     prob = prepare_problem(level=3, use_quotient=True,
#                            use_spin_blocks=True)
#
#     results = {}
#     nus = [Fraction(a, 32) for a in range(20, 33)]   # 0.625 .. 1.0
#     for nu in nus:
#         lo, _ = solve_at_filling(prob, nu)
#         results[str(nu)] = float(lo)
#         with open("nu_scan.json", "w") as fh:
#             json.dump(results, fh, indent=1)
#         log(f"BANKED nu={nu} ({float(nu):.4f}): {lo:.6f}")
#
#     pts = sorted((Fraction(s), v) for s, v in results.items())
#     log("\nnu        bound        second difference")
#     for a in range(1, len(pts) - 1):
#         (x0, y0), (x1, y1), (x2, y2) = pts[a-1], pts[a], pts[a+1]
#         d2 = (y2 - y1) / float(x2 - x1) - (y1 - y0) / float(x1 - x0)
#         log(f"{float(x1):.4f}   {y1:+.6f}   {d2:+.6f}")
#     log("\nconvex check: all second differences must be >= 0 up to "
#         "solver noise (~1e-5); a run of near-zero d2 = linear segment "
#         "= Maxwell/phase-separation signal, refine locally at /128.")

# if __name__ == "__main__":
#     results = {}
#     def bank(key, val):
#         results[key] = val
#         with open("spin_test.json", "w") as fh:
#             json.dump(results, fh, indent=1)
#         log(f"BANKED {key} = {val:.6f}")
#
#     set_patch(3)
#     log("=== A: P=3 L3, quotient, NO spin blocks (certified path) ===")
#     probA = prepare_problem(level=3, use_quotient=True,
#                             use_spin_blocks=False)
#     loA, _ = solve_at_filling(probA, Fraction(7, 8))
#     bank("A_no_spin", loA)
#     sizesA = sorted((bd[0] for bd in probA["blockdata"]), reverse=True)
#     del probA
#
#     log("\n=== B: P=3 L3, quotient + spin blocks (must match exactly) ===")
#     probB = prepare_problem(level=3, use_quotient=True,
#                             use_spin_blocks=True)
#     loB, _ = solve_at_filling(probB, Fraction(7, 8))
#     bank("B_spin", loB)
#     sizesB = sorted((bd[0] for bd in probB["blockdata"]), reverse=True)
#
#     log(f"\nA (no spin): {loA:.6f}   blocks top-5 {sizesA[:5]}")
#     log(f"B (spin):    {loB:.6f}   blocks top-5 {sizesB[:5]}")
#     log(f"delta: {abs(loA - loB):.2e}   (solver tolerance expected; "
#         f"both must be -0.938438)")