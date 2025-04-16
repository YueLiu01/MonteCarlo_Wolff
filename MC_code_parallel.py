# ─── 0.  Imports ────────────────────────────────────────────────────────────────
from concurrent.futures import ProcessPoolExecutor, as_completed   # NEW

import numpy as np
from numba import njit

@njit(nopython=True)
def chord(x,L):
    '''
    chord distance
    '''
    return L/np.pi*np.sin(np.pi*x/L)

@njit(nopython=True)
def build_nn_neighbors(Lx, Ly):
    """
    Build a lookup table of the 4 nearest neighbors (with periodic BCs)
    for each site (x, y) in an Lx x Ly lattice.

    neighbors[x, y, k, 0..1], where k in {0,1,2,3}:
      k=0 -> x+1 (up in x-direction, mod Lx)
      k=1 -> x-1 (down in x-direction, mod Lx)
      k=2 -> y+1 (right in y-direction, mod Ly)
      k=3 -> y-1 (left in y-direction, mod Ly)
    """
    neighbors = np.empty((Lx, Ly, 4, 2), dtype=np.int32)
    for x in range(Lx):
        x_up   = (x + 1) % Lx
        x_down = (x - 1) % Lx
        for y in range(Ly):
            y_right = (y + 1) % Ly
            y_left  = (y - 1) % Ly

            # Up in x-direction
            neighbors[x, y, 0, 0] = x_up
            neighbors[x, y, 0, 1] = y

            # Down in x-direction
            neighbors[x, y, 1, 0] = x_down
            neighbors[x, y, 1, 1] = y

            # Right in y-direction
            neighbors[x, y, 2, 0] = x
            neighbors[x, y, 2, 1] = y_right

            # Left in y-direction
            neighbors[x, y, 3, 0] = x
            neighbors[x, y, 3, 1] = y_left
    return neighbors

@njit(nopython=True)
def build_defect_line_arrays(Ly, K):
    """
    Precompute the 1/r^2 couplings along the horizontal defect line in x-direction.

    We'll store, for each y in [0..Ly-1]:
       line_y2[y, idx] = y2
       line_J[y, idx]  = K / (|y2 - y|^2)

    skipping y2 == y, hence each row has (Ly-1) entries.
    
    If you want periodic distance in y, use:
       dy = min(abs(y2 - y), Ly - abs(y2 - y))
    instead of simple abs(y2 - y).
    """
    line_y2 = np.empty((Ly, Ly - 1), dtype=np.int32)
    line_J  = np.empty((Ly, Ly - 1), dtype=np.float64)
    for y in range(Ly):
        idx = 0
        for y2 in range(Ly):
            if y2 == y:
                continue
            dy0 = abs(y2 - y)
            dist_ring = min(dy0, Ly - dy0)
            dy = chord(dist_ring, Ly)
            val = K / (dy**2)
            line_y2[y, idx] = y2
            line_J[y, idx]  = val
            idx += 1
    return line_y2, line_J

@njit(nopython=True)
def wolff_update(spins, Lx, Ly, beta, J, defect_x, line_y2, line_J, neighbors):
    """
    Perform one Wolff cluster update on an Lx x Ly lattice.
    'defect_x' = row (in x) for the horizontal defect line.
    'line_y2' and 'line_J': the 1/r^2 couplings among y-values on that line.
    'neighbors': array with shape (Lx, Ly, 4, 2) for nearest neighbors.
    """
    # 1) Pick a random seed (x0, y0)
    # x0 = np.random.randint(Lx)
    # y0 = np.random.randint(Ly)
    # seed_spin = spins[x0, y0]
    
    if np.random.random() < 0.5:
        # Choose from the defect line only
        x0 = defect_x
        y0 = np.random.randint(Ly)
    else:
        # Choose from anywhere
        x0 = np.random.randint(Lx)
        y0 = np.random.randint(Ly)
    seed_spin = spins[x0, y0]

    # Track which sites are in the cluster
    cluster_mask = np.zeros((Lx, Ly), dtype=np.bool_)
    cluster_mask[x0, y0] = True

    # Stack of site indices (encoded as idx = x*Ly + y)
    stack = np.empty(Lx * Ly, dtype=np.int32)
    top = 0
    stack[top] = x0 * Ly + y0
    top += 1

    while top > 0:
        top -= 1
        site_idx = stack[top]
        x = site_idx // Ly
        y = site_idx % Ly

        # --- Nearest neighbors (4 directions) ---
        for k in range(4):
            nx = neighbors[x, y, k, 0]
            ny = neighbors[x, y, k, 1]
            if (not cluster_mask[nx, ny]) and (spins[nx, ny] == seed_spin):
                p_bond = 1.0 - np.exp(-2.0 * beta * J)
                if np.random.random() < p_bond:
                    cluster_mask[nx, ny] = True
                    stack[top] = nx * Ly + ny
                    top += 1

        # --- Defect line bonds ---
        # If we are on x == defect_x, bond to other y in that row with 1/r^2
        if x == defect_x:
            for idx_y2 in range(Ly - 1):
                y2 = line_y2[y, idx_y2]
                J_long = line_J[y, idx_y2]
                if (not cluster_mask[x, y2]) and (spins[x, y2] == seed_spin):
                    p_bond = 1.0 - np.exp(-2.0 * beta * J_long)
                    if np.random.random() < p_bond:
                        cluster_mask[x, y2] = True
                        stack[top] = x * Ly + y2
                        top += 1

    # 3) Flip all spins in the cluster
    for idx2 in range(Lx * Ly):
        xx = idx2 // Ly
        yy = idx2 % Ly
        if cluster_mask[xx, yy]:
            spins[xx, yy] = -spins[xx, yy]

@njit(nopython=True)
def run_single_chain(Lx, Ly, J, K, beta, n_sweeps, save_spins):
    """
    Runs one Wolff chain on an Lx x Ly system for n_sweeps.

    If save_spins=True, we build a 3D array spin_history[sweep, x, y]
    storing the spin configuration after each sweep.

    Returns:
      final_spins (Lx x Ly),
      final_magnetization (float),
      spin_history (n_sweeps x Lx x Ly) or a dummy shape if not saving.
    """
    defect_x = Lx // 2

    neighbors = build_nn_neighbors(Lx, Ly)
    line_y2, line_J = build_defect_line_arrays(Ly, K)

    # Initialize spins randomly
    spins = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(Lx, Ly))

    if save_spins:
        spin_history = np.zeros((n_sweeps, Lx, Ly), dtype=np.int8)
    else:
        # Dummy array with shape (1,1,1)
        spin_history = np.zeros((1, 1, 1), dtype=np.int8)
    mag_history = np.zeros(n_sweeps, dtype=np.float64)
    defect_mag_history = np.zeros(n_sweeps, dtype=np.float64)
    for sweep in range(n_sweeps):
        wolff_update(spins, Lx, Ly, beta, J, defect_x, line_y2, line_J, neighbors)
        if save_spins:
            spin_history[sweep] = spins
        mag_history[sweep] = np.mean(spins)
        defect_mag_history[sweep] = np.mean(spins[Lx//2,:])

    # final_m = np.mean(spins)
    return spins, mag_history, defect_mag_history, spin_history

def example_run(Lx=8, Ly=16, J=1.0, K=0.5, beta=0.44, n_sweeps=500, save_spins=True):
    """
    High-level example for a Lx x Ly lattice with one horizontal defect line.
    By default, saves the spin history to 'spin_history.npy'.
    """
    final_spins, mag_history, defect_mag_history, spin_history = run_single_chain(
        Lx, Ly, J, K, beta, n_sweeps, save_spins=save_spins
    )

    # print(f"Final magnetization = {final_m:.3f}")
    # if save_spins:
    #     # Save the entire spin history after the simulation
    #     np.save("spin_history.npy", spin_history)
        # print("Spin history saved to 'spin_history.npy'.")

    return mag_history, defect_mag_history, spin_history


# ─── 1.  A *single‑K* worker ----------------------------------------------------
def simulate_one_K(K, Lx, Ly, J, beta, n_sweeps):
    """
    Run one Monte‑Carlo chain for a given K and return the post‑processed data
    you were printing inside the loop.
    """
    mag_hist, defect_hist, _ = example_run(
        Lx=Lx, Ly=Ly, J=J, K=K, beta=beta,
        n_sweeps=n_sweeps, save_spins=False
    )

    # discard first 5 000 sweeps (thermalisation)
    m_slice      = mag_hist[5000:]**2
    defect_slice = defect_hist[5000:]**2

    M      = np.abs(m_slice).mean()
    Merr   = np.std(np.abs(m_slice)) / np.sqrt(n_sweeps)
    M_def  = np.abs(defect_slice).mean()
    Mderr  = np.std(defect_slice)    / np.sqrt(n_sweeps)

    return K, M, Merr, M_def, Mderr        # everything you need later




# ─── 2.  Parallel outer loop over K_list ---------------------------------------
N_K   = 64
L_list = [16, 32, 64, 128, 256, 512, 1024]
beta   = 0.4406867935097715
n_sweeps = 40000
data_total = []
data_defect = []
for Ly in L_list:
    Lx = Ly
    J  = 1.0

    K_list = np.linspace(0, 0.5, N_K) / beta

    M_list, Merr_list = [], []
    M_defect_list, M_defect_err_list = [], []

    print("\nbeta*K, M, Merr, M_defect, M_defect_err")

    # Use every core (change max_workers to limit)
    with ProcessPoolExecutor(max_workers=8) as pool:
        # launch one process per K
        futures = [pool.submit(simulate_one_K, K, Lx, Ly, J, beta, n_sweeps)
                   for K in K_list]

        # gather results in *completion* order (fastest first)
        for fut in as_completed(futures):
            K, M, Merr, M_def, Mderr = fut.result()

            # store in the same order as K_list  -------------------------------
            idx = np.searchsorted(K_list, K)
            M_list.insert(idx, M)
            Merr_list.insert(idx, Merr)
            M_defect_list.insert(idx, M_def)
            M_defect_err_list.insert(idx, Mderr)

            print(K*beta, M, Merr, M_def, Mderr)

    data_total.append(M_list)
    data_defect.append(M_defect_list)

# ─── 3.  Save to disk (unchanged) ----------------------------------------------
np.save("Isingdefect_alpha2_Mag_sq_total_betac_1024.npy", data_total)
np.save("Isingdefect_alpha2_Mag_sq_defect_betac_1024.npy", data_defect)

