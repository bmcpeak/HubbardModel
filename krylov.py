"""
krylov.py -- symmetry-channelized, RADIUS-CAPPED H-selected bases.

Design: adapt a small symmetric seed into irrep channels (d4, two_s,
N-charge) once (hw lift + D4 projection), then iterate O -> [H, O] within
each channel, TRUNCATING each image to Chebyshev radius RHO about the
center. With the cap this is no longer a true Krylov filtration -- images
are capped-H-selected operators; moment positivity over them is valid for
any operator set, but <[H,O]> = 0 equalities are harvested from the TRUE
(uncapped) commutator forms only, and only when in-span (the capped
operator's commutator identity would be a different, wrong statement).

Fixes in this revision, each aimed at a measured cost:
  * RHO cap on images        -- the 644 -> 18,528 variable explosion
  * gcd rescale per image    -- ||H||^k bignum coefficient growth
  * _assert_pure sampled     -- ~70 group sweeps per image, verified
                                exhaustively at depths 1-2 already
Purity note under the cap: truncation at a D4-symmetric radius about the
center commutes with the point group and with S^+ (both preserve radius
and act site-local), so capped images remain channel-pure; the sampled
assert remains the tripwire for that argument.
"""

import time
import random
from fractions import Fraction
from math import gcd

from algebra import (add, multiply, dagger, commutator,
                     to_linear_form, normalize_form, support_sites,
                     sites_of, unpack, pack_sites, parity_sign,
                     c_op, d_op, IDENTITY, sz_charge, n_charge,
                     W, UP, DN)
from physics import ad_H_op, local_spin_charge
from symmetry import PG8, _char, _D4_CHARS

CX, CY = W // 2, W // 2
RHO = 2                      # Chebyshev support cap for images
PURITY_SAMPLE = 0.1          # fraction of images spot-checked
_rng = random.Random(11)

# ============================================================
# Group action on operator dicts
# ============================================================

def act_op(op, pg_index, sf):
    M = PG8[pg_index]
    out = {}
    for key, v in op.items():
        D, C = unpack(key)
        def f(p):
            i, j, s = p
            u, w_ = i - CX, j - CY
            return (M[0][0] * u + M[0][1] * w_ + CX,
                    M[1][0] * u + M[1][1] * w_ + CY, s ^ sf)
        Ds = [f(p) for p in sites_of(D)]
        Cs = [f(p) for p in sites_of(C)]
        sgn = parity_sign(Ds) * parity_sign(Cs)
        k2 = pack_sites(Ds, Cs)
        out[k2] = out.get(k2, 0) + sgn * v
    return {k: v for k, v in out.items() if v != 0}

# ============================================================
# Spin raising / hw lift
# ============================================================

def _op_sites(op):
    sites = set()
    for k in op:
        sites |= support_sites(k)
    return sites

def sp_raise(op):
    sites = _op_sites(op)
    if not sites:
        return {}
    return commutator(local_spin_charge(sites, +1), op)

def raise_to_hw(op, max_steps=8):
    cur = op
    for _ in range(max_steps):
        up = sp_raise(cur)
        if not up:
            return cur
        cur = up
    raise AssertionError("hw lift did not terminate: check seed")

# ============================================================
# D4 projection, purity, rescale, cap
# ============================================================

def _clear_denominators(raw):
    if not raw:
        return {}
    den = 1
    for x in raw.values():
        den = den * x.denominator // gcd(den, x.denominator)
    return {k: int(x * den) for k, x in raw.items()}

def d4_project(op, d4name):
    raw = {}
    for pg in range(8):
        w = (PG8[pg][0][0] * 2) if d4name == "E" else _char(d4name, pg)
        if w == 0:
            continue
        img = act_op(op, pg, 0)
        for k, v in img.items():
            raw[k] = raw.get(k, Fraction(0)) + Fraction(w, 8) * v
    return _clear_denominators({k: v for k, v in raw.items() if v != 0})

def _assert_pure(op, d4name, context):
    for other in _D4_CHARS:
        if other == d4name:
            continue
        assert not d4_project(op, other), \
            f"{context}: operator leaked from {d4name} into {other}"
    assert not sp_raise(op), f"{context}: operator lost highest weight"

def rescale(op):
    """Divide by the gcd of coefficients (pure rescaling; spans, PSD
    content, and channel labels unchanged)."""
    if not op:
        return op
    g = 0
    for v in op.values():
        g = gcd(g, abs(v))
    return op if g <= 1 else {k: v // g for k, v in op.items()}

def cap_radius(op, rho=RHO):
    """Truncate to monomials supported within Chebyshev radius rho of
    (CX, CY). D4-symmetric region about the center, so channel purity
    survives truncation."""
    out = {}
    for k, v in op.items():
        if all(max(abs(i - CX), abs(j - CY)) <= rho
               for (i, j) in support_sites(k)):
            out[k] = v
    return out

# ============================================================
# Seed adaptation
# ============================================================

def adapt_seed(seed_ops, log=print):
    channels = {}
    for op in seed_ops:
        qs = {n_charge(k) for k in op}
        assert len(qs) == 1, "seed operator mixes N-charge"
        q = qs.pop()
        hw = op if op == IDENTITY else raise_to_hw(op)
        ms = {sz_charge(k) for k in hw}
        assert len(ms) == 1, "hw lift produced mixed Sz: seed not pure"
        two_s = ms.pop()
        for d4name in _D4_CHARS:
            p = d4_project(hw, d4name)
            if not p:
                continue
            _assert_pure(p, d4name, "seed adaptation")   # always, cheap
            channels.setdefault((d4name, two_s, q), []).append(rescale(p))
    for ch in list(channels):
        channels[ch] = _independent(channels[ch])
        if not channels[ch]:
            del channels[ch]
    for ch, ops in sorted(channels.items()):
        log(f"    seed channel {ch}: {len(ops)} ops")
    return channels

# ============================================================
# Rank dedup and capped towers
# ============================================================

def _independent(ops):
    basis, pivots, keep = [], [], []
    for op in ops:
        v = {k: Fraction(x) for k, x in op.items()}
        for bvec, p in zip(basis, pivots):
            if p in v and v[p] != 0:
                f = v[p] / bvec[p]
                for k, x in bvec.items():
                    v[k] = v.get(k, Fraction(0)) - f * x
                    if v[k] == 0:
                        del v[k]
        if v:
            basis.append(v)
            pivots.append(min(v))
            keep.append(op)
    return keep

def krylov_channel(ch, seed_ops, depth, t, U, log=print):
    d4name = ch[0]
    tower = list(seed_ops)
    frontier = list(seed_ops)
    for step in range(depth):
        new = []
        for op in frontier:
            if not _op_sites(op):
                continue                        # identity
            img = ad_H_op(op, t, U)
            img = rescale(cap_radius(img))
            if not img:
                continue
            if _rng.random() < PURITY_SAMPLE:
                _assert_pure(img, d4name, f"{ch} depth {step+1}")
            new.append(img)
        before = len(tower)
        tower = _independent(tower + new)
        frontier = tower[before:]
        log(f"    {ch} depth {step+1}: +{len(frontier)} "
            f"(total {len(tower)})")
        if not frontier:
            break
    return tower

# ============================================================
# Channel moment blocks; equality harvest (TRUE commutators)
# ============================================================

def channel_blockdata(tower):
    n = len(tower)
    dags = [dagger(o) for o in tower]
    a0 = ([], [], [])
    pv = {}
    for i in range(n):
        for j in range(i + 1):
            form = to_linear_form(multiply(dags[i], tower[j]))
            for k, v in form.items():
                tgt = a0 if k == 0 else pv.setdefault(k, ([], [], []))
                tgt[0].append(i); tgt[1].append(j)
                tgt[2].append(float(v))
    return (n, a0, pv)

def harvest_equalities(towers, t, U, span, log=print):
    """<[H, O]> = 0 from the TRUE commutator of each tower element (the
    capped operator IS the tower element, so its true commutator is the
    valid statement about it), kept when in-span."""
    seen, out = set(), []
    oos = 0
    for tower in towers:
        for op in tower:
            if not _op_sites(op):
                continue
            form = to_linear_form(ad_H_op(op, t, U))
            if not form:
                continue
            if not set(form) <= span:
                oos += 1
                continue
            h = normalize_form(form)
            if h not in seen:
                seen.add(h)
                out.append(form)
    log(f"    krylov equalities: {len(out)} kept, {oos} out-of-span")
    return out

# ============================================================
# Full build
# ============================================================

def build_krylov_problem(seed_ops, depth, t, U, log=print):
    t0 = time.time()
    channels = adapt_seed(seed_ops, log=log)
    blockdata, chan_index, towers = [], [], []
    variables = set()
    for ch, ops in sorted(channels.items()):
        tower = krylov_channel(ch, ops, depth, t, U, log=log)
        towers.append(tower)
        blockdata.append(channel_blockdata(tower))
        chan_index.append(ch)
        variables.update(blockdata[-1][2].keys())
    for k in variables:
        assert n_charge(k) == 0, "N-charged variable escaped"
    span = variables | {0}
    equalities = harvest_equalities(towers, t, U, span, log=log)
    log(f"krylov build: {time.time()-t0:.1f}s, RHO={RHO}, "
        f"{len(blockdata)} channels, sizes {[b[0] for b in blockdata]}, "
        f"{len(variables)} raw vars, {len(equalities)} equalities")
    return blockdata, variables, equalities, chan_index

# ============================================================
# Seeds
# ============================================================

def seed_single_site():
    r = (CX, CY)
    nu = multiply(d_op(*r, UP), c_op(*r, UP))
    nd = multiply(d_op(*r, DN), c_op(*r, DN))
    n_minus = add(nu, {k: -v for k, v in nd.items()})
    return [IDENTITY, add(nu, nd), n_minus, multiply(nu, nd),
            c_op(*r, UP), c_op(*r, DN), d_op(*r, UP), d_op(*r, DN)]

def seed_star(include_exchange=True):
    r = (CX, CY)
    rx = (CX + 1, CY)
    ops = list(seed_single_site())
    ops += [c_op(*rx, UP), c_op(*rx, DN), d_op(*rx, UP), d_op(*rx, DN)]
    nu_x = multiply(d_op(*rx, UP), c_op(*rx, UP))
    nd_x = multiply(d_op(*rx, DN), c_op(*rx, DN))
    ops += [add(nu_x, nd_x),
            add(nu_x, {k: -v for k, v in nd_x.items()}),
            multiply(nu_x, nd_x)]
    hop_u = add(multiply(d_op(*r, UP), c_op(*rx, UP)),
                multiply(d_op(*rx, UP), c_op(*r, UP)))
    hop_d = add(multiply(d_op(*r, DN), c_op(*rx, DN)),
                multiply(d_op(*rx, DN), c_op(*r, DN)))
    ops += [add(hop_u, hop_d),
            add(hop_u, {k: -v for k, v in hop_d.items()})]
    if include_exchange:
        ops.append(multiply(
            multiply(d_op(*r, UP), c_op(*r, DN)),
            multiply(d_op(*rx, DN), c_op(*rx, UP))))
    return ops