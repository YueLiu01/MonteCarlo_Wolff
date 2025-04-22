"""
Optimised Wolff Monte‑Carlo with a 1/r^2 defect line
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

import numpy as np
from numba import njit, prange


@njit(inline="always")
def chord(x: int, L: int) -> float:
    """Chord distance on a ring of circumference *L* (periodic)."""
    return L / np.pi * np.sin(np.pi * x / L)


@njit(inline="always")
def nn(x: int, y: int, direction: int, Lx: int, Ly: int) -> Tuple[int, int]:
    """Return the neighbour of (x, y) in *direction* (0:+x, 1:‑x, 2:+y, 3:‑y)."""
    if direction == 0:
        return (x + 1) % Lx, y
    elif direction == 1:
        return (x - 1) % Lx, y
    elif direction == 2:
        return x, (y + 1) % Ly
    else:
        return x, (y - 1) % Ly


@njit(nopython=True)
def build_defect_line_arrays(Ly: int, K: float):
    """Pre‑compute the 1/r^2 couplings along the defect line.

    Returns (*uint16* indices, *float32* couplings).
    """
    line_y2 = np.empty((Ly, Ly - 1), dtype=np.uint16)
    line_J = np.empty((Ly, Ly - 1), dtype=np.float32)

    for y in range(Ly):
        idx = 0
        for y2 in range(Ly):
            if y2 == y:
                continue
            dy0 = abs(y2 - y)
            dist_ring = min(dy0, Ly - dy0)
            dy = chord(dist_ring, Ly)
            val = K / (dy * dy)

            line_y2[y, idx] = np.uint16(y2)
            line_J[y, idx] = np.float32(val)
            idx += 1
    return line_y2, line_J


@njit(nopython=True)
def wolff_update(
    spins: np.ndarray,  # (Lx, Ly) int8
    beta: float,
    J: float,
    defect_x: int,
    line_y2: np.ndarray,  # (Ly, Ly‑1) uint16
    line_J: np.ndarray,  # (Ly, Ly‑1) float32
):
    """One Wolff cluster update with long‑range bonds on a defect line."""

    Lx, Ly = spins.shape

    # ‑‑‑ choose a seed site --------------------------------------------------
    if np.random.random() < 0.5:
        x0 = defect_x
        y0 = np.random.randint(Ly)
    else:
        x0 = np.random.randint(Lx)
        y0 = np.random.randint(Ly)

    seed_spin = spins[x0, y0]

    # cluster mask and DFS stack (start small, grow if needed)
    cluster = np.zeros((Lx, Ly), dtype=np.bool_)
    cluster[x0, y0] = True

    stack = np.empty(4 * Ly, dtype=np.uint32)  # initial capacity
    top = 0
    stack[top] = x0 * Ly + y0
    top += 1

    p_nn = 1.0 - np.exp(-2.0 * beta * J)  # nearest‑neighbour bond prob.

    while top:
        top -= 1
        idx = stack[top]
        x = idx // Ly
        y = idx - x * Ly

        # local 4 neighbours --------------------------------------------------
        for k in range(4):
            nx, ny = nn(x, y, k, Lx, Ly)
            if (not cluster[nx, ny]) and (spins[nx, ny] == seed_spin):
                if np.random.random() < p_nn:
                    cluster[nx, ny] = True
                    if top == stack.size:
                        stack = np.resize(stack, stack.size * 2)
                    stack[top] = nx * Ly + ny
                    top += 1

        # long‑range bonds on the defect line --------------------------------
        if x == defect_x:
            for j in range(Ly - 1):
                y2 = int(line_y2[y, j])
                if (not cluster[x, y2]) and (spins[x, y2] == seed_spin):
                    p_bond = 1.0 - np.exp(-2.0 * beta * line_J[y, j])
                    if np.random.random() < p_bond:
                        cluster[x, y2] = True
                        if top == stack.size:
                            stack = np.resize(stack, stack.size * 2)
                        stack[top] = x * Ly + y2
                        top += 1

    # flip the cluster --------------------------------------------------------
    for idx in range(cluster.size):
        if cluster.ravel()[idx]:
            spins.ravel()[idx] = -spins.ravel()[idx]


@njit(nopython=True)
def run_single_chain(
    Lx: int,
    Ly: int,
    J: float,
    K: float,
    beta: float,
    n_sweeps: int,
):
    """Run *n_sweeps* Wolff updates and return the magnetisation history."""

    defect_x = Lx // 2

    line_y2, line_J = build_defect_line_arrays(Ly, K)

    spins = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(Lx, Ly))

    m_hist = np.empty(n_sweeps, dtype=np.float32)
    m_def_hist = np.empty(n_sweeps, dtype=np.float32)

    for sweep in range(n_sweeps):
        wolff_update(spins, beta, J, defect_x, line_y2, line_J)
        m_hist[sweep] = np.mean(spins)
        m_def_hist[sweep] = np.mean(spins[defect_x])

    return m_hist, m_def_hist


def simulate_one_K(K: float, L: int, beta: float, n_sweeps: int):
    """Wrapper to launch a *single* Monte‑Carlo chain for coupling *K*."""
    m_hist, m_def_hist = run_single_chain(L, L, 1.0, K, beta, n_sweeps)

    # drop first 20 % as thermalisation (example)
    start = n_sweeps // 10
    m2 = (m_hist[start:] ** 2).mean()
    m2_err = (m_hist[start:] ** 2).std(ddof=1) / np.sqrt(n_sweeps - start)

    m2_def = (m_def_hist[start:] ** 2).mean()
    m2_def_err = (
        (m_def_hist[start:] ** 2).std(ddof=1) / np.sqrt(n_sweeps - start)
    )

    return K*beta, m2, m2_err, m2_def, m2_def_err


def main():
    L = 8192 # lattice size (square)
    beta_c = 0.4406867935097715  # critical beta for 2D Ising

    n_sweeps = 180000  # number of sweeps per K
    K_list = np.arange(32)*0.5/63/ beta_c

    results = []
    with ProcessPoolExecutor(max_workers=16) as pool:  # adjust workers
        futures = [
            pool.submit(simulate_one_K, K, L, beta_c, n_sweeps) for K in K_list
        ]
        for fut in as_completed(futures):
            results.append(fut.result())
            print("K*beta,  <m²>, err   | defect <m²>, err  =>", results[-1])

    # sort by K and save
    results.sort(key=lambda t: t[0])
    np.save("ising_defect_results_8192_a.npy", np.array(results))


if __name__ == "__main__":
    main()

