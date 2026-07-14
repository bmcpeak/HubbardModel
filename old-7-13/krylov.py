"""
krylov.py -- UNCAPPED Hankel-Krylov towers with layered identification.

Replaces the radius-capped H-selected scheme. Design changes and why:

1. NO RADIUS CAP. Towers are true Liouvillian orbits P_b^k = Lv^k s_b,
   Lv = [H, .], computed ONCE per (channel, seed) up to k = 2K (+1 with
   Stieltjes). The cap existed to control the variable explosion of
   expanding <T_i^dag T_j>; the Hankel rewriting removes that expansion
   entirely, so the cap's reason to exist is gone -- and with it the
   broken Liouvillian symmetry, the O(n^2) entry count, and the
   cap-vs-Ward bookkeeping.

2. HANKEL BY CONSTRUCTION. The shuffle identity
       <(Lv^i s_a)^dag Lv^j s_b> = <s_a^dag Lv^{i+j} s_b>   (mod Ward)
   is hard-wired: entry ((a,i),(b,j)) IS the form G_{i+j}^{ab}. Product
   Ward identities <[H, T_i^dag T_j]> = 0 are therefore consumed
   structurally -- there is nothing to harvest and nothing for the
   solver to eliminate. Validity does not depend on which Wards run.py
   imposes: in the true ground state every rewritten entry literally
   equals <T_i^dag T_j>, so the block is PSD there, which is all a
   relaxation needs.

3. LAYERED IDENTIFICATION (the ChatGPT-Y hybrid). Each G_k^{ab} is
   expanded into canonical monomial variables while (a) the tower
   element fits in TOWER_CAP monomials and (b) the resulting linear
   form fits in FORM_BUDGET terms. Past either budget the entry becomes
   an ABSTRACT scalar Y[(ch,a,b,k)] -- still constrained by the Hankel
   and Stieltjes blocks, no longer tied to the monomial pool. k_id is
   recorded per (channel, seed pair) and reported. Abstract-only mode
   is FORM_BUDGET = 0; fully-identified mode is FORM_BUDGET = inf.

4. STIELTJES BLOCKS. For each channel we also emit the shifted block
       M1[(a,i),(b,j)] = G_{i+j+1}^{ab}  >= 0,
   i.e. <A^dag (H - E0) B> >= 0 on the tower span: the ground-state
   positivity of the Liouvillian spectral measure. NOTE: this is a
   GROUND-STATE-ONLY condition (excited stationary states violate it).
   It is not implied by stationarity Wards and is expected to carry a
   large share of the spectral information. Toggle: STIELTJES.

5. NO TOWER DEDUP. _independent is retained ONLY for seed adaptation
   (tiny, exact). Linearly dependent tower rows just make the PSD block
   singular, which is harmless; the depth-cubed exact-elimination
   hotspot is deleted rather than optimized.

6. COEFFICIENT HANDLING -- read this before "optimizing". Tower
   elements are stored gcd-rescaled with the cumulative scale recorded
   as a float (lam[k]); the exact value of Lv^k s is lam[k] * P~^k.
   The scale is multiplied back AT EMISSION. Do NOT rescale tower
   elements without recording the scale: an entry h_{i+j}/g_{i+j} with
   unfactorized per-k scales is NOT a diagonal congruence of a Hankel
   matrix and silently destroys PSD-validity of the rewriting.
   Emission-side conditioning uses the factorized congruence
   D_{a,i} D_{b,j} with D from tower coefficient norms, which IS valid.

Returned blockdata keeps the (n, a0, pv) COO convention. New:
  * blocks are labeled by chan_index entries (ch, kind) with kind in
    {"H0", "H1"}; there are up to 2 blocks per channel.
  * the variable pool may contain ABSTRACT keys: tuples
    ("Y", ci, a, b, k). run.py must treat these as plain SDP scalars:
    skip n_charge assertions and skip the monomial Ward-quotient
    substitution for them (they are already quotiented by construction).
  * equalities are only the vanishing-row forms <Lv^k s_b> = 0 (k >= 1,
    neutral channel; identified layer only -- in the abstract layer the
    corresponding Y is pinned to zero at build time instead).
"""

import time
import random
from fractions import Fraction
from math import gcd, sqrt

from algebra import (add, multiply, dagger, commutator,
                     to_linear_form, normalize_form, support_sites,
                     sites_of, unpack, pack_sites, parity_sign,
                     c_op, d_op, IDENTITY, sz_charge, n_charge,
                     W, UP, DN)
from physics import ad_H_op, local_spin_charge
from symmetry import PG8, _char, _D4_CHARS

CX, CY = W // 2, W // 2

# ---------------- configuration ----------------
K_DEPTH = 3            # Hankel depth: block rows i = 0..K per seed
STIELTJES = True       # emit shifted (ground-state) blocks
FORM_BUDGET = 40_000   # max #monomials in an identified entry form
TOWER_CAP = 400_000    # max #monomials in a stored tower element
PURITY_SAMPLE = 0.05   # fraction of tower elements spot-checked (cheap check)
_rng = random.Random(11)

# ============================================================
# Group action, hw lift, D4 projection, seed adaptation
# (unchanged from the capped version except the purity tripwire)
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

def _cheap_purity(op, two_s, context):
    """Tripwire replacing the 56-sweep projector suite: Sz-homogeneity
    (one pass) plus highest-weight (one commutator). D4 purity of true
    Liouvillian images is a theorem given seed purity ([H, .] commutes
    with the point group), so it is checked exhaustively only at seed
    time via _assert_pure_full."""
    ms = {sz_charge(k) for k in op}
    assert ms == {two_s}, f"{context}: Sz mixed {ms}"
    assert not sp_raise(op), f"{context}: lost highest weight"

def _assert_pure_full(op, d4name, context):
    for other in _D4_CHARS:
        if other == d4name:
            continue
        assert not d4_project(op, other), \
            f"{context}: operator leaked from {d4name} into {other}"
    assert not sp_raise(op), f"{context}: operator lost highest weight"

def rescale_tracked(op):
    """gcd rescale returning (rescaled_op, scale). Exact: op == scale*out."""
    if not op:
        return op, 1
    g = 0
    for v in op.values():
        g = gcd(g, abs(v))
    if g <= 1:
        return op, 1
    return {k: v // g for k, v in op.items()}, g

def _independent(ops):
    """Exact rank filter -- SEED ADAPTATION ONLY (a handful of small
    operators). Towers are not deduplicated at all; see module docstring."""
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
            _assert_pure_full(p, d4name, "seed adaptation")
            g0, _ = rescale_tracked(p)
            channels.setdefault((d4name, two_s, q), []).append(g0)
    for ch in list(channels):
        channels[ch] = _independent(channels[ch])
        if not channels[ch]:
            del channels[ch]
    for ch, ops in sorted(channels.items()):
        log(f"    seed channel {ch}: {len(ops)} ops")
    return channels

# ============================================================
# Uncapped towers with tracked scales
# ============================================================

def _support_radius(op):
    r = 0
    for k in op:
        for (i, j) in support_sites(k):
            r = max(r, abs(i - CX), abs(j - CY))
    return r

# usable radius from the center before hop construction leaves the
# window (min over the four margins; W even makes the center off-axis)
_R_AVAIL = min(CX, CY, W - 1 - CX, W - 1 - CY)

def liouville_tower(seed, kmax, ch, t, U, log=print):
    """P[k] ~ Lv^k seed, stored gcd-rescaled; lam[k] float such that the
    exact operator is lam[k] * P[k]. Stops early (marking truncation)
    if a tower element exceeds TOWER_CAP monomials OR if the next
    Liouvillian application would leave the lattice window (support
    radius + 1 for the hop partner must fit in _R_AVAIL). Beyond the
    truncation the channel's entries fall back to the abstract layer."""
    P, lam = [dict(seed)], [1.0]
    for k in range(1, kmax + 1):
        if _support_radius(P[-1]) + 1 > _R_AVAIL:
            log(f"      {ch}: tower truncated at k={k} by the lattice "
                f"window (support radius {_support_radius(P[-1])}, "
                f"usable {_R_AVAIL}); raise W in algebra.py for full "
                f"identification")
            break
        img = ad_H_op(P[-1], t, U)
        if not img:
            break
        if len(img) > TOWER_CAP:
            log(f"      {ch}: tower truncated at k={k} "
                f"({len(img)} monomials > TOWER_CAP)")
            break
        if _rng.random() < PURITY_SAMPLE:
            _cheap_purity(img, ch[1], f"{ch} tower k={k}")
        img, g = rescale_tracked(img)
        P.append(img)
        lam.append(lam[-1] * g)
    return P, lam

# ============================================================
# Entry emission: identified form or abstract scalar
# ============================================================

class _YPool:
    """Abstract-scalar registry. Keys ('Y', ci, a, b, k) are canonical
    in (a, b): the (b, a) entry reuses the (a, b) scalar (moments real:
    real Hamiltonian, real ground state, integer-coefficient seeds --
    Y_k^{ab} = conj(Y_k^{ba}) = Y_k^{ba})."""
    def __init__(self):
        self.keys = set()
    def key(self, ci, a, b, k):
        a, b = min(a, b), max(a, b)
        kk = ("Y", ci, a, b, k)
        self.keys.add(kk)
        return kk

def _entry_form(sa_dag, P_b, lam_b, k):
    """Identified linear form of s_a^dag * Lv^k s_b, or None if over
    budget (tower truncated short of k, or form too large)."""
    if k >= len(P_b):
        return None
    prod = multiply(sa_dag, P_b[k])
    form = to_linear_form(prod)
    if len(form) > FORM_BUDGET:
        return None
    if lam_b[k] != 1.0:
        form = {m: v * lam_b[k] for m, v in form.items()}
    return form

def _emit_block(ci, ch, seeds, towers, lams, K, shift, ypool,
                variables, identity_flags, vanishing, dscale):
    """COO block in the (n, a0, pv) convention for entries
    G_{i+j+shift}^{ab}, i,j = 0..K per seed. shift=0: Hankel; shift=1:
    Stieltjes. Entries are pre-scaled by the factorized congruence
    D_{a,i} D_{b,j} (valid: positive diagonal congruence)."""
    r = len(seeds)
    n = r * (K + 1)
    a0 = ([], [], [])
    pv = {}
    dags = [dagger(s) for s in seeds]

    # cache: entry content depends only on (a, b, k). Each cached value
    # is one of:
    #   ("zero", None)       -- exactly the zero operator (identity col)
    #   ("ward", form|None)  -- identity row, k >= 1: vanishing Ward;
    #                           form None -> abstract layer, pinned to 0
    #   ("form", form|None)  -- generic; form None -> abstract scalar
    cache = {}
    def G(a, b, k):
        key = (a, b, k)
        if key not in cache:
            if identity_flags[b] and k >= 1:
                cache[key] = ("zero", None)       # Lv^k 1 = 0 exactly
            elif identity_flags[a] and k >= 1:
                f = _entry_form(dags[a], towers[b], lams[b], k)
                cache[key] = ("ward", f)          # <Lv^k s_b> = 0 on-shell
                if f:
                    vanishing.append(dict(f))
            else:
                # mirror reuse: G^{ab}_k and G^{ba}_k agree on-shell and
                # run.py mirrors the lower triangle, so one form serves
                # both orientations. This also prevents an identified
                # entry from coexisting with an untied abstract mirror
                # (a strength leak when towers truncate asymmetrically),
                # and halves the pair-multiply work.
                mirror = cache.get((b, a, k))
                if mirror is not None and mirror[0] == "form" \
                        and mirror[1] is not None:
                    cache[key] = mirror
                else:
                    f = _entry_form(dags[a], towers[b], lams[b], k)
                    if f is None and b != a:
                        f = _entry_form(dags[b], towers[a], lams[a], k)
                    cache[key] = ("form", f)
        return cache[key]

    for a in range(r):
        for i in range(K + 1):
            I = a * (K + 1) + i
            for b in range(r):
                for j in range(K + 1):
                    J = b * (K + 1) + j
                    if J > I:
                        continue          # lower triangle, run.py mirrors
                    k = i + j + shift
                    kind, f = G(a, b, k)
                    if kind == "zero":
                        continue
                    s = dscale[(a, i)] * dscale[(b, j)]
                    if f is not None:
                        for m, v in f.items():
                            tgt = a0 if m == 0 else \
                                pv.setdefault(m, ([], [], []))
                            tgt[0].append(I); tgt[1].append(J)
                            tgt[2].append(float(v) * s)
                            if m != 0:
                                variables.add(m)
                    elif kind == "ward":
                        continue          # abstract vanishing row: hard 0
                    else:
                        yk = ypool.key(ci, a, b, k)
                        tgt = pv.setdefault(yk, ([], [], []))
                        tgt[0].append(I); tgt[1].append(J)
                        tgt[2].append(1.0 * s)
                        variables.add(yk)
    return (n, a0, pv)

# ============================================================
# Full build
# ============================================================

def build_krylov_problem(seed_ops, depth, t, U, log=print):
    """depth == K (Hankel block has K+1 rows per seed). Towers reach
    k = 2K (+1 if STIELTJES)."""
    t0 = time.time()
    K = depth
    kmax = 2 * K + (1 if STIELTJES else 0)
    seed_r = max((_support_radius(s) for s in seed_ops if s != IDENTITY),
                 default=0)
    need = seed_r + kmax + 1
    if need > _R_AVAIL:
        log(f"  NOTE: full towers need usable radius {need} but the "
            f"window gives {_R_AVAIL} (W = {W}); towers will truncate "
            f"and deep entries go abstract. For full identification "
            f"set W >= {2 * need + 1} (odd) in algebra.py.")
    channels = adapt_seed(seed_ops, log=log)

    blockdata, chan_index = [], []
    variables = set()
    vanishing = []
    ypool = _YPool()
    report = {}

    for ci, (ch, seeds) in enumerate(sorted(channels.items())):
        towers, lams, identity_flags, dscale = [], [], [], {}
        for a, s in enumerate(seeds):
            is_id = (s == IDENTITY)
            identity_flags.append(is_id)
            if is_id:
                towers.append([dict(IDENTITY)])   # Lv 1 = 0
                lams.append([1.0])
            else:
                P, lam = liouville_tower(s, kmax, ch, t, U, log=log)
                towers.append(P)
                lams.append(lam)
            # factorized congruence scales from tower coefficient norms
            for i in range(K + 1):
                kk = min(2 * i, len(towers[a]) - 1)
                nrm = max((abs(v) for v in towers[a][kk].values()),
                          default=1) * max(lams[a][kk], 1.0)
                dscale[(a, i)] = 1.0 / sqrt(max(nrm, 1e-300))

        blk = _emit_block(ci, ch, seeds, towers, lams, K, 0, ypool,
                          variables, identity_flags, vanishing, dscale)
        blockdata.append(blk)
        chan_index.append((ch, "H0"))
        if STIELTJES:
            blk1 = _emit_block(ci, ch, seeds, towers, lams, K, 1, ypool,
                               variables, identity_flags, vanishing,
                               dscale)
            blockdata.append(blk1)
            chan_index.append((ch, "H1"))

        # per-channel identification report
        kid = {}
        for a in range(len(seeds)):
            reach = len(towers[a]) - 1
            kid[a] = reach
        report[ch] = dict(seeds=len(seeds), tower_reach=kid,
                          sizes=[len(p) for p in towers[0]])

    # dedup vanishing forms
    seen, equalities = set(), []
    for f in vanishing:
        h = normalize_form(f)
        if h not in seen:
            seen.add(h)
            equalities.append(f)

    for k in variables:
        if not isinstance(k, tuple):      # monomial keys only
            assert n_charge(k) == 0, "N-charged variable escaped"

    n_abs = len(ypool.keys)
    n_mono = len(variables) - n_abs
    log(f"krylov build: {time.time()-t0:.1f}s, K={K}, "
        f"{len(blockdata)} blocks "
        f"({'H0+H1' if STIELTJES else 'H0'}), "
        f"sizes {[b[0] for b in blockdata]}, "
        f"{n_mono} monomial vars + {n_abs} abstract Y, "
        f"{len(equalities)} vanishing-row equalities")
    for ch, r in sorted(report.items()):
        log(f"    {ch}: towers reach k={r['tower_reach']}, "
            f"element sizes {r['sizes']}")
    return blockdata, variables, equalities, chan_index

# ============================================================
# Seeds (unchanged)
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
