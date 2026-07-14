#!/usr/bin/env python3
"""
Ground-state LOWER bounds for the anharmonic oscillator by SDP, comparing
two operator bases at equal depth.

    H = p^2 + x^2 + g x^4,   [x, p] = i   (hbar = 1;  E0 ~ 1.3923516 at g=1)

Formulation (deliberately the same as the Hubbard pipeline):
    minimize  <H>
    over moment functionals L subject to
      * moment-matrix PSD over an operator basis,
      * stationarity Ward identities  <[H, w]> = 0, harvested for every
        normal-ordered monomial w whose image lies in the variable span,
      * CCR / Hermiticity relations  L(w^dag) = conj(L(w)),
      * parity (the ground state of an even potential is parity-even),
      * <1> = 1.
The true ground state satisfies every constraint, so the optimum is a
rigorous lower bound for ANY choice of operator basis.

Basis A ("degree", the patch-basis analog): all normal-ordered words
    x^a p^b with a + b <= n.
NOTE: the pure {x^k} basis is NOT used: in this min-<H> formulation its
moment matrix contains no p-content, the uncertainty principle never
enters an inequality, and the bound is trivial. The x^k basis of the
bootstrap literature lives in the eigenstate-scan formulation, where
<x^k H> = E <x^k> injects the quantum term. Words in {x, p} are the
honest analog of the Hubbard patch basis.

Basis B ("krylov"): towers T_i = Lv^i s, Lv = [H, .], in two parity
channels: seed {x} (odd) and seeds {1, x^2} (even). Moment-matrix
entries are REWRITTEN with the shuffle identity (valid in any
stationary state, i.e. on the Ward surface -- and exactly true in the
ground state, which is what validity requires):

    <(Lv^i s_a)^dag  Lv^j s_b>  =  <s_a  Lv^{i+j}  s_b>

so each channel block is HANKEL: only 2L+1 distinct entry forms
G_k = s_a Lv^k s_b instead of (L+1)^2 products, and the variable span is
the (much smaller) set of monomials appearing in the G_k. The vanishing
rows <Lv^k s> = 0 (k >= 1) are imposed explicitly as operator-level Ward
identities. A raw (unrewritten) Krylov variant is also assembled for the
comparison table.

Moment parametrization: real ground state  =>  <x^a p^b> = (-i)^b r_ab
with r real; parity  =>  r_ab = 0 for a+b odd. Complex Hermitian PSD
blocks are imposed through the real embedding [[A, -B], [B, A]].
"""

import sys
from math import comb, factorial

import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G_COUPLING = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
DEGREE_RANGE = list(range(1, 9))      # words x^a p^b with a+b <= n
KRYLOV_RANGE = list(range(0, 7))      # tower depth L
SOLVERS = ["MOSEK", "CLARABEL", "SCS"]
OUTPNG = "anharmonic_bootstrap.png"

# ======================================================================
# Operator algebra on normal-ordered words  x^a p^b  (dict (a,b)->complex)
# ======================================================================

def prune(A, tol=1e-13):
    if not A:
        return {}
    mx = max(abs(v) for v in A.values())
    return {k: v for k, v in A.items() if abs(v) > tol * mx}

def op_add(A, B, cB=1.0):
    out = dict(A)
    for k, v in B.items():
        out[k] = out.get(k, 0j) + cB * v
    return prune(out)

def mono_mul(m1, m2):
    """(x^a1 p^b1)(x^a2 p^b2) -> normal order.
    Uses  p^b x^a = sum_k (-i)^k k! C(b,k) C(a,k) x^{a-k} p^{b-k}."""
    (a1, b1), (a2, b2) = m1, m2
    out = {}
    for k in range(min(b1, a2) + 1):
        c = ((-1j) ** k) * factorial(k) * comb(b1, k) * comb(a2, k)
        out[(a1 + a2 - k, b1 + b2 - k)] = c
    return out

def op_mul(A, B):
    out = {}
    for m1, c1 in A.items():
        for m2, c2 in B.items():
            for m3, c3 in mono_mul(m1, m2).items():
                out[m3] = out.get(m3, 0j) + c1 * c2 * c3
    return prune(out)

def op_dag(A):
    """(c x^a p^b)^dag = conj(c) p^b x^a, normal-ordered."""
    out = {}
    for (a, b), c in A.items():
        for k in range(min(a, b) + 1):
            ck = ((-1j) ** k) * factorial(k) * comb(a, k) * comb(b, k)
            m = (a - k, b - k)
            out[m] = out.get(m, 0j) + np.conj(c) * ck
    return prune(out)

def H_op(g):
    return {(0, 2): 1.0 + 0j, (2, 0): 1.0 + 0j, (4, 0): g + 0j}

def ad_H(A, g):
    Hd = H_op(g)
    return op_add(op_mul(Hd, A), op_mul(A, Hd), -1.0)

def op_norm(A):
    return max((abs(v) for v in A.values()), default=1.0)

# ======================================================================
# Moment forms:  L(op) as a complex-linear form over real vars r_ab
# ======================================================================

def form(A):
    """dict (a,b) -> complex coeff, meaning L(A) = sum coeff * r_ab.
    Applies the phase (-i)^b and drops odd-parity monomials (moment 0)."""
    f = {}
    for (a, b), c in A.items():
        if (a + b) % 2:
            continue
        f[(a, b)] = f.get((a, b), 0j) + c * ((-1j) ** b)
    return prune(f)

def scale_form(f, s):
    return {k: v * s for k, v in f.items()}

def form_sub(f1, f2):
    out = dict(f1)
    for k, v in f2.items():
        out[k] = out.get(k, 0j) - v
    return prune(out)

# ======================================================================
# Basis builders  ->  list of blocks; each block is a matrix of forms
# ======================================================================

def build_degree_blocks(n):
    """All words x^a p^b, a+b <= n, split into parity blocks (the cross
    block vanishes identically by parity)."""
    words = [{(a, d - a): 1.0 + 0j} for d in range(n + 1) for a in range(d + 1)]
    blocks = []
    for par in (0, 1):
        ws = [w for w in words if sum(next(iter(w))) % 2 == par]
        dags = [op_dag(w) for w in ws]
        blk = [[form(op_mul(dags[i], ws[j])) for j in range(len(ws))]
               for i in range(len(ws))]
        if ws:
            blocks.append(blk)
    return blocks, []

def liouville_powers(seed, K, g):
    seq = [dict(seed)]
    for _ in range(K):
        seq.append(ad_H(seq[-1], g))
    return seq

def build_krylov_blocks(L, g, raw=False):
    """Two channels: odd (seed x), even (seeds 1, x^2).
    rewritten: entry((a,i),(b,j)) = form(s_a * Lv^{i+j} s_b)   [Hankel]
    raw:       entry = form(T_i^dag T_j) with T = actual tower elements.
    Returns (blocks, extra_ward_forms). The extra forms are the
    operator-level identities <Lv^k x^2> = 0, k >= 1 (the vanishing row);
    their monomials are added to the span by the assembler."""
    sx = {(1, 0): 1.0 + 0j}
    s2 = {(2, 0): 1.0 + 0j}
    K = 2 * L
    Px = liouville_powers(sx, K, g)
    P2 = liouville_powers(s2, K, g)

    blocks = []
    if raw:
        # ---- odd channel: basis {T_i(x)} ----
        T = Px[:L + 1]
        D = [1.0 / op_norm(t) for t in T]
        dags = [op_dag(t) for t in T]
        blk = [[scale_form(form(op_mul(dags[i], T[j])), D[i] * D[j])
                for j in range(L + 1)] for i in range(L + 1)]
        blocks.append(blk)
        # ---- even channel: basis {1} + {T_i(x^2)} ----
        T = [{(0, 0): 1.0 + 0j}] + P2[:L + 1]
        D = [1.0 / op_norm(t) for t in T]
        dags = [op_dag(t) for t in T]
        N = L + 2
        blk = [[scale_form(form(op_mul(dags[i], T[j])), D[i] * D[j])
                for j in range(N)] for i in range(N)]
        blocks.append(blk)
    else:
        # ---- odd channel, Hankel entries x * Lv^{i+j} x ----
        nx = [op_norm(p) for p in Px]
        D = [1.0 / np.sqrt(max(nx[2 * i], 1e-300)) for i in range(L + 1)]
        blk = [[scale_form(form(op_mul(sx, Px[i + j])), D[i] * D[j])
                for j in range(L + 1)] for i in range(L + 1)]
        blocks.append(blk)
        # ---- even channel: basis {1, (x^2, i)} ----
        n2 = [op_norm(p) for p in P2]
        D = [1.0] + [1.0 / np.sqrt(max(n2[2 * i], 1e-300))
                     for i in range(L + 1)]
        N = L + 2
        blk = [[None] * N for _ in range(N)]
        for I in range(N):
            for J in range(N):
                if I == 0 and J == 0:
                    f = {(0, 0): 1.0 + 0j}
                elif I == 0 or J == 0:
                    k = max(I, J) - 1          # cross entries <1 * Lv^k x^2>
                    f = form(P2[k])
                else:
                    f = form(op_mul(s2, P2[(I - 1) + (J - 1)]))
                blk[I][J] = scale_form(f, D[I] * D[J])
        blocks.append(blk)

    # vanishing-row Wards <Lv^k s> = 0 (odd-channel ones are parity-empty)
    extra = [form(P2[k]) for k in range(1, K + 1)]
    extra = [f for f in extra if f]
    return blocks, extra

# ======================================================================
# Assembly and solve
# ======================================================================

def _dfact(k):
    """(k-1)!! for even k >= 0."""
    out = 1.0
    for j in range(1, k, 2):
        out *= j
    return out

def _sigma(a, b, vx=0.5, vp=0.5):
    """Rough magnitude of <x^a p^b> from a Gaussian estimate; used purely
    as a variable scaling (does not change the mathematical program)."""
    return np.sqrt(_dfact(2 * a) * vx ** a * _dfact(2 * b) * vp ** b)

def assemble_and_solve(blocks, extra_ward_forms, g, label=""):
    # ---- variable span ----
    span = set()
    for B in blocks:
        for row in B:
            for f in row:
                span |= set(f)
    span |= set(form(H_op(g)))
    for f in extra_ward_forms:
        span |= set(f)
    span.add((0, 0))
    # close under dagger descendants (a-k, b-k), needed by Hermiticity rows
    stack = list(span)
    while stack:
        a, b = stack.pop()
        for k in range(1, min(a, b) + 1):
            m = (a - k, b - k)
            if m not in span:
                span.add(m)
                stack.append(m)
    mono = sorted(span)
    idx = {m: i for i, m in enumerate(mono)}
    nv = len(mono)
    # variable scaling: true moment r_ab = sig_ab * (solver variable)
    sig = np.array([_sigma(a, b) for (a, b) in mono])
    r = cp.Variable(nv)

    cons = [r[idx[(0, 0)]] == 1]          # sig(0,0) = 1
    eq_rows = []          # numeric rows for the effective-DOF rank count

    def add_form_eq(f):
        if not f:
            return
        fs = {m: v * sig[idx[m]] for m, v in f.items()}
        mx = max(abs(v) for v in fs.values())
        fs = {m: v / mx for m, v in fs.items()}
        for part in ("real", "imag"):
            row = {m: getattr(v, part) for m, v in fs.items()
                   if abs(getattr(v, part)) > 1e-11}
            if row:
                cons.append(
                    cp.sum(cp.hstack([c * r[idx[m]] for m, c in row.items()]))
                    == 0)
                vec = np.zeros(nv)
                for m, c in row.items():
                    vec[idx[m]] = c
                eq_rows.append(vec)

    # ---- CCR / Hermiticity:  L(w^dag) = conj(L(w)) ----
    for (a, b) in mono:
        if b == 0:
            continue
        lhs = form(op_dag({(a, b): 1.0 + 0j}))
        rhs = {(a, b): np.conj((-1j) ** b)}
        add_form_eq(form_sub(lhs, rhs))

    # ---- operator-level Wards passed in (Krylov vanishing rows) ----
    for f in extra_ward_forms:
        add_form_eq(f)

    # ---- monomial Ward harvest: <[H, w]> = 0 when in-span ----
    dmax = max(a + b for (a, b) in mono)
    nward = 0
    for d in range(0, dmax + 1, 2):
        for a in range(d + 1):
            f = form(ad_H({(a, d - a): 1.0 + 0j}, g))
            if f and set(f) <= span:
                add_form_eq(f)
                nward += 1

    # ---- PSD blocks via real embedding of the Hermitian part ----
    sizes = []
    for B in blocks:
        N = len(B)
        sizes.append(N)
        mats = {}
        for i in range(N):
            for j in range(N):
                for m, c in B[i][j].items():
                    if m not in mats:
                        # store per-variable complex coefficient matrix
                        mats[m] = np.zeros((N, N), complex)
                    mats[m][i, j] += c
        if not mats:
            continue
        expr = None
        for m, C in mats.items():
            C = C * sig[idx[m]]
            E = np.block([[C.real, -C.imag], [C.imag, C.real]])
            E = (E + E.T) / 2                 # Hermitian part, embedded
            term = r[idx[m]] * E
            expr = term if expr is None else expr + term
        cons.append(expr >> 0)

    # ---- objective:  <H> = -r_{0,2} + r_{2,0} + g r_{4,0} ----
    fH = form(H_op(g))
    obj = cp.sum(cp.hstack([v.real * sig[idx[m]] * r[idx[m]]
                            for m, v in fH.items()]))
    prob = cp.Problem(cp.Minimize(obj), cons)

    used = None
    for accept in (("optimal",), ("optimal", "optimal_inaccurate")):
        for s in SOLVERS:
            if s not in cp.installed_solvers():
                continue
            try:
                prob.solve(solver=s, verbose=False)
            except Exception:
                continue
            if prob.status in accept:
                used = s
                break
        if used:
            break

    if eq_rows:
        A = np.vstack(eq_rows + [np.eye(nv)[idx[(0, 0)]]])
        rank = np.linalg.matrix_rank(A, tol=1e-9)
    else:
        rank = 1
    dof = nv - rank
    return dict(bound=prob.value, nvars=nv, dof=dof, sizes=sizes,
                nward=nward, solver=used, status=prob.status,
                maxdeg=dmax, label=label)

# ======================================================================
# Exact reference (Rayleigh-Ritz in a large oscillator basis)
# ======================================================================

def exact_E0(g, N=500):
    a = np.diag(np.sqrt(np.arange(1, N)), 1)
    x = (a + a.T) / np.sqrt(2)
    p = 1j * (a.T - a) / np.sqrt(2)
    Hm = p @ p + x @ x + g * np.linalg.matrix_power(x, 4)
    Hm = (Hm + Hm.conj().T) / 2
    return float(np.linalg.eigvalsh(Hm.real)[0])

# ======================================================================
# Main
# ======================================================================

def main():
    g = G_COUPLING
    e0 = exact_E0(g)
    print(f"g = {g},  exact E0 = {e0:.10f}\n")

    # quick self-test: harmonic limit should give E >= 1 (attained)
    blocks, extra = build_degree_blocks(2)
    chk = assemble_and_solve(blocks, extra, 0.0, "selftest")
    assert chk["bound"] is not None and abs(chk["bound"] - 1.0) < 1e-5, chk
    print(f"self-test (g=0, n=2): bound = {chk['bound']:.8f}  "
          f"(expect 1.0)  PASS\n")

    hdr = (f"{'run':<22}{'depth':>6}{'blocks':>12}{'maxdeg':>7}"
           f"{'vars':>6}{'dof':>5}{'#ward':>6}{'bound':>14}{'gap':>11}")
    print(hdr)
    print("-" * len(hdr))

    deg_pts, kry_pts = [], []
    for n in DEGREE_RANGE:
        blocks, extra = build_degree_blocks(n)
        res = assemble_and_solve(blocks, extra, g, f"degree n={n}")
        deg_pts.append((n, res))
        print(f"{'degree words<=n':<22}{n:>6}{str(res['sizes']):>12}"
              f"{res['maxdeg']:>7}{res['nvars']:>6}{res['dof']:>5}"
              f"{res['nward']:>6}{res['bound']:>14.8f}"
              f"{e0 - res['bound']:>11.2e}")

    for L in KRYLOV_RANGE:
        blocks, extra = build_krylov_blocks(L, g, raw=False)
        res = assemble_and_solve(blocks, extra, g, f"krylov L={L}")
        kry_pts.append((L, res))
        print(f"{'krylov (Hankel rw)':<22}{L:>6}{str(res['sizes']):>12}"
              f"{res['maxdeg']:>7}{res['nvars']:>6}{res['dof']:>5}"
              f"{res['nward']:>6}{res['bound']:>14.8f}"
              f"{e0 - res['bound']:>11.2e}")

    print()
    for L in KRYLOV_RANGE:
        blocks, extra = build_krylov_blocks(L, g, raw=True)
        res = assemble_and_solve(blocks, extra, g, f"krylov raw L={L}")
        print(f"{'krylov (raw prods)':<22}{L:>6}{str(res['sizes']):>12}"
              f"{res['maxdeg']:>7}{res['nvars']:>6}{res['dof']:>5}"
              f"{res['nward']:>6}{res['bound']:>14.8f}"
              f"{e0 - res['bound']:>11.2e}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = [n for n, _ in deg_pts]
    ys = [r["bound"] for _, r in deg_pts]
    ax.plot(xs, ys, "o-", color="tab:blue",
            label=r"degree basis: words $x^ap^b$, $a{+}b\leq n$  (vs $n$)")
    for n, res in deg_pts:
        ax.annotate(f"{res['dof']}", (n, res["bound"]),
                    textcoords="offset points", xytext=(0, -13),
                    ha="center", fontsize=7, color="tab:blue")
    xs = [L for L, _ in kry_pts]
    ys = [r["bound"] for _, r in kry_pts]
    ax.plot(xs, ys, "s-", color="tab:red",
            label=r"Krylov tower, Hankel-rewritten  (vs depth $L$)")
    for L, res in kry_pts:
        ax.annotate(f"{res['dof']}", (L, res["bound"]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color="tab:red")
    ax.axhline(e0, ls="--", c="k", lw=1,
               label=f"exact $E_0$ = {e0:.7f}")
    ax.set_xlabel(r"depth parameter ($n$ for degree basis, $L$ for Krylov)")
    ax.set_ylabel(r"SDP lower bound on $E_0$")
    ax.set_title(rf"$H=p^2+x^2+gx^4$, $g={g}$"
                 "\n(point labels: effective DOF after equalities)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPNG, dpi=150)
    print(f"\nplot saved to {OUTPNG}")

if __name__ == "__main__":
    main()