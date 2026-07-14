"""Exact free-fermion (U=0) reference moments via Wick's theorem.

Purpose: the strongest end-to-end validation of the assembled SDP. A
closed-shell free-fermion Slater state on a large L x L torus is an exact
many-body state possessing every symmetry the moment layer assumes
(translations, D4, spin flip, reality), and all its moments reduce to
determinants of the two-point function. Its moment vector must therefore
be exactly feasible for the assembled problem — any sign error anywhere
in algebra/symmetry/moments/assembly shows up as an infeasible exact
state. It also provides a variational upper reference at any U:
E_var = e_kin + U * <n_up n_dn>_free, which every valid lower bound must
sit below.

Wick conventions (pinned by tests/test_bootstrap_level2.py against
independent applier-based ED, spinless and spinful):

  * canonical per-spin string  d_{i1}..d_{ip} c_{jq}..c_{j1}  (i asc, j asc)
    has expectation det[ g(i_a, j_b) ]_{a,b}  — the canonical ordering
    (annihilators reversed) is exactly the one that makes the determinant
    sign-free;
  * mixed-spin monomials factorize into (up part)(down part) times the
    crossing sign (-1)^{# pairs (down-op before up-op) in the string},
    since opposite-spin fermion operators anticommute.

Closed shells: partially filled degenerate momentum shells would break D4
invariance of the Slater state and falsely fail the feasibility test, so
the filling snaps to the largest complete-shell count at or below the
target; the achieved density is reported and must be used as the SDP
filling in the feasibility check.
"""

from __future__ import annotations

import numpy as np

from algebra import bits_ascending, bits_descending
from modes import Lattice


class TorusGaussianState:
    """Gaussian (free-fermion) state on an L x L torus at EXACT per-spin
    density nu_sigma: fill all shells strictly below the Fermi shell, and
    occupy every mode of the (degenerate) Fermi shell with the equal
    fraction f in [0, 1] that hits the target exactly.

    Uniform shell occupation keeps the state translation-, D4-, flip-, and
    reality-invariant; a legitimate density matrix, Gaussian, so Wick's
    theorem applies with the same contraction g. Its moment vector is
    therefore exactly feasible for the assembled SDP at the exact filling,
    and its energy e_kin + U <n_up><n_dn> upper-bounds the same convex
    envelope the SDP lower-bounds.
    """

    def __init__(self, L: int, nu_sigma: float, t: float = 1.0):
        kx = 2 * np.pi * np.arange(L) / L
        eps = -2 * t * (np.cos(kx)[:, None] + np.cos(kx)[None, :])
        target = nu_sigma * L * L
        if not (0 <= target <= L * L):
            raise ValueError("filling out of range")
        flat = np.sort(eps.ravel())
        # shell containing the Fermi level
        n_full = int(np.floor(target))
        # extend to shell boundaries
        lo = n_full
        while lo > 0 and flat[lo] - flat[lo - 1] < 1e-10:
            lo -= 1
        hi = n_full
        while hi < L * L and (hi == 0 or flat[hi] - flat[hi - 1] < 1e-10):
            hi += 1
        # occupations: 1 below the shell, f on the shell
        shell_size = hi - lo
        f = (target - lo) / shell_size if shell_size else 0.0
        assert -1e-12 <= f <= 1 + 1e-12
        e_shell = flat[lo] if lo < L * L else None
        nk = np.where(eps < (flat[lo] - 1e-10 if lo > 0 else -np.inf), 1.0, 0.0)
        if shell_size and lo < L * L:
            nk = nk + np.where(np.abs(eps - flat[lo]) < 1e-10, f, 0.0)
        assert abs(nk.sum() - target) < 1e-8
        self.L = L
        self.nu_sigma = nu_sigma
        self.fermi_fraction = f
        # g(dx, dy) = (1/L^2) sum_k n(k) e^{i k . d}; real by D4 symmetry
        self._g = np.real(np.fft.ifft2(nk.astype(complex)))
        self.e_kin_per_site_per_spin = float((eps * nk).sum()) / (L * L)

    def g(self, dx: int, dy: int) -> float:
        return float(self._g[dx % self.L, dy % self.L])


# backward-compatible alias (closed-shell behaviour is the f in {0,1} case)
TorusFreeFermions = TorusGaussianState


class WickEvaluator:
    """Evaluate canonical-monomial moments in a spin-symmetric Slater state.

    g2(site_a, site_b) -> <d_a c_b> for a single spin species; sites are
    (x, y). For the torus state use lambda a, b: tor.g(a[0]-b[0], a[1]-b[1]);
    for OBC ED validation pass the correlation matrix as a callable.
    """

    def __init__(self, lat: Lattice, g2):
        self.lat = lat
        self.g2 = g2

    def _string_tokens(self, m):
        """(spin, site) tokens in canonical string order."""
        lat = self.lat
        toks = []
        for i in bits_ascending(m[0]):
            x, y, s = lat.decode(i)
            toks.append((s, "d", (x, y)))
        for j in bits_descending(m[1]):
            x, y, s = lat.decode(j)
            toks.append((s, "c", (x, y)))
        return toks

    def mono_expval(self, m) -> float:
        if m == (0, 0):
            return 1.0
        if self.lat.charges(m) != (0, 0):
            return 0.0
        toks = self._string_tokens(m)
        # crossing sign: (-1)^{# (down before up) pairs}
        crossings = 0
        ups_seen_after = 0
        for s, _, _ in reversed(toks):
            if s == 0:
                ups_seen_after += 1
            else:
                crossings += ups_seen_after
        sign = -1.0 if crossings & 1 else 1.0
        val = sign
        for spin in (0, 1):
            ds = [site for (s, kind, site) in toks if s == spin and kind == "d"]
            cs_string = [site for (s, kind, site) in toks if s == spin and kind == "c"]
            # string order for c's is descending; det convention wants
            # ascending index order j1 < ... < jq
            cs = list(reversed(cs_string))
            if not ds and not cs:
                continue
            G = np.array([[self.g2(a, b) for b in cs] for a in ds])
            val *= float(np.linalg.det(G)) if len(ds) else 1.0
        return val

    def moment_vector(self, table) -> np.ndarray:
        y = np.zeros(len(table))
        for k, m in enumerate(table.monos):
            y[k] = self.mono_expval(m)
        return y

    def double_occupancy(self) -> float:
        """<n_up n_dn> on one site = g(0,0)^2 for the uncorrelated state."""
        return self.g2((0, 0), (0, 0)) ** 2
