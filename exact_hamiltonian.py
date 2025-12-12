"""
** written by Claude Opus 4.5 **
Hubbard Model - Exact Diagonalization
Reproduces the Mathematica code

Local basis: |0>, |down>, |up>, |up,down>  (indices 0, 1, 2, 3)
Convention: |up,down> = c^dagger_up c^dagger_down |0>
"""

import numpy as np
from functools import reduce

# ===== Local 4x4 operators =====

ddown_local = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 0]
], dtype=float)

dup_local = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, -1, 0, 0]
], dtype=float)

cdown_local = ddown_local.T
cup_local = dup_local.T
id_local = np.eye(4)
P_local = np.diag([1., -1., -1., 1.])


def kron_list(op_list):
    """Tensor product of list of operators."""
    return reduce(np.kron, op_list)


def c(site, spin, nsite):
    """Annihilation operator c_{site, spin}. Site is 1-indexed."""
    local_op = cup_local if spin == 'up' else cdown_local
    ops = [P_local if j < site else (local_op if j == site else id_local)
           for j in range(1, nsite + 1)]
    return kron_list(ops)


def d(site, spin, nsite):
    """Creation operator c^dagger_{site, spin}. Site is 1-indexed."""
    local_op = dup_local if spin == 'up' else ddown_local
    ops = [P_local if j < site else (local_op if j == site else id_local)
           for j in range(1, nsite + 1)]
    return kron_list(ops)


def hubbard_hamiltonian(nsite, t, U):
    """
    H = -t sum_{i, sigma} (c^dag_i c_{i+1} + h.c.) + U sum_i n_i,up n_i,down
    Open boundary conditions.
    """
    dim = 4 ** nsite
    H = np.zeros((dim, dim))

    # Hopping
    for i in range(1, nsite):
        for spin in ['up', 'down']:
            H -= t * (d(i, spin, nsite) @ c(i + 1, spin, nsite) +
                      d(i + 1, spin, nsite) @ c(i, spin, nsite))

    # Repulsion
    for i in range(1, nsite + 1):
        n_up = d(i, 'up', nsite) @ c(i, 'up', nsite)
        n_down = d(i, 'down', nsite) @ c(i, 'down', nsite)
        H += U * n_up @ n_down

    return H


def number_operator(nsite):
    """Total particle number operator."""
    dim = 4 ** nsite
    N = np.zeros((dim, dim))
    for i in range(1, nsite + 1):
        for spin in ['up', 'down']:
            N += d(i, spin, nsite) @ c(i, spin, nsite)
    return N


def project_to_sector(H, nvals, n_particles):
    """Extract Hamiltonian block for fixed particle number."""
    idx = np.where(nvals == n_particles)[0]
    return H[np.ix_(idx, idx)], idx


# ===== Main =====

if __name__ == "__main__":
    # Parameters
    nsite = 2
    t_val = 1.0
    U_val = 4.0

    print(f"Hubbard model: {nsite} sites, t = {t_val}, U = {U_val}")
    print("=" * 50)

    # Build Hamiltonian
    H = hubbard_hamiltonian(nsite, t_val, U_val)
    dim = 4 ** nsite

    # Number operator
    N = number_operator(nsite)
    nvals = np.diag(N).astype(int)
    print(f"Particle numbers by basis state: {nvals}")

    # Full spectrum
    evals_full = np.sort(np.linalg.eigvalsh(H))
    print(f"\nFull spectrum ({dim} states):")
    print(evals_full)

    # Project to N=2 sector (half-filling)
    H2, idx2 = project_to_sector(H, nvals, n_particles=2)
    print(f"\nN=2 sector indices: {idx2}")
    print(f"N=2 sector dimension: {len(idx2)}")

    # Diagonalize N=2 sector
    evals2, evecs2 = np.linalg.eigh(H2)
    print(f"\nN=2 spectrum:")
    print(evals2)

    # Ground state
    E0 = evals2[0]
    psi0 = evecs2[:, 0]

    print(f"\nGround state energy: {E0}")
    print(f"Expected: 2 - 2√2 = {2 - 2 * np.sqrt(2):.10f}")

    print(f"\nGround state (in N=2 sector basis):")
    print(psi0)

    # Normalize for clarity
    print(f"\nGround state (rescaled):")
    psi0_rescaled = psi0 / psi0[0] if abs(psi0[0]) > 1e-10 else psi0
    print(psi0_rescaled)

    # Verify: H|psi> = E|psi>
    residual = np.linalg.norm(H2 @ psi0 - E0 * psi0)
    print(f"\nVerification ||H*psi - E*psi|| = {residual:.2e}")