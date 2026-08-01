import lattice_utilities as lat_utils
import numpy as np
import wigner

# Parameters
L = 8
n_sweeps = 5000

A = 1.0
B = np.exp(-np.sqrt(2))
C = np.exp(-2)
V = 1.0
J = 1.0

V_list = np.array([A, A, A, A, B, B, B, B, C, C, C, C], dtype=np.float64) * V
J_list = np.array([A, A, A, A, B, B, B, B, C, C, C, C], dtype=np.float64) * J 
schedule = np.arange(0.2, 0, -0.0025, dtype=np.float64)

particle_lattice = lat_utils.generate_filled_square_lattice(L, filling = 0.75)

print("Starting Anneal...")
wigner.wigner_anneal(particle_lattice, 3, n_sweeps, V_list, J_list, schedule)