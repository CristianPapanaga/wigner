# cython: language_level = 3
# cython: boundscheck = False
# cython: cdivision = True
# cython: wraparound = False
# cython: initializedcheck = False

import numpy as np
cimport numpy as np

from libc.math cimport exp, sqrt

import lattice_utilities as lat_utils # type: ignore
import data_utilities as dat_utils

# Silence MSVC implicit truncation warning C4244 globally within this module.
cdef extern from *:
    """
    #if defined(_MSC_VER)
    #pragma warning(disable : 4244)
    #endif
    """

cpdef void wigner_anneal(np.ndarray[np.int64_t, ndim = 1, mode = "c"] particle_lattice, int nth_neighbours, long n_sweeps,
                         np.ndarray[np.float_t, ndim = 1, mode = "c"] V_list, np.ndarray[np.float_t, ndim = 1, mode = "c"] J_list,
                         np.ndarray[np.float_t, ndim = 1, mode = "c"] schedule, double thermal_frac = 0.1, double data_collection_interval = 1):

    '''
    Perform simulated annealing of the lattice following a provided cooling schedule ie. slowly cool the lattice from a high temperature
    through a series of temperatures in the list schedule to some final temperature. The number of sweeps done per scheduled
    temperature is n_sweeps.

    Annealing helps prevent the Monte Carlo method becoming stuck in local minima when looking for the global minimum (ie. ground state).
    '''

    cdef np.int64_t N, L, i, j, k, i_idx, j_idx, nn, n_neighbours
    cdef long n_steps, n_thermalisation_steps, data_collection_counter, data_collection_step, thermalisation_step, step
    cdef double E, deltaE, V, J, T
    cdef list E_array, M_array, E_arrays, M_arrays, data_names
    cdef np.ndarray[np.int64_t, ndim = 1, mode = "c"] randints_filled, randints_empty, filled_coords, empty_coords, nn_i, nn_j
    cdef np.ndarray[np.int64_t, ndim = 2, mode = "c"] lattice_nn
    cdef np.ndarray[np.float_t, ndim = 1, mode = "c"] randfloats, i_spin_initial, i_spin_candidate, i_spin_diff
    cdef np.ndarray[np.float_t, ndim = 2, mode = "c"] randgauss, spin_lattice, randspins

    # Fast temporary registers for explicit 3D vector operations
    cdef double sx_init, sy_init, sz_init
    cdef double sx_cand, sy_cand, sz_cand
    cdef double sdx, sdy, sdz
    cdef double tmp_x, tmp_y, tmp_z

    N = <np.int64_t>particle_lattice.shape[0]
    L = <np.int64_t>sqrt(<double>N)
    n_neighbours = <np.int64_t>nth_neighbours * 4
    lattice_nn = lat_utils.get_nearest_neighbour_coords(L, nth_neighbours)

    n_steps = n_sweeps * N 
    n_thermalisation_steps = <long>(n_steps * thermal_frac)
    data_collection_counter = 0 # Keeps track of how many steps it has been since data was last collected.
    data_collection_step = <long>(data_collection_interval * N) # Step at which data is collected.
    thermalisation_step = n_thermalisation_steps - data_collection_step

    filled_coords = np.array([idx for idx, val in enumerate(particle_lattice) if val == 1], dtype=np.int64)
    empty_coords = np.array([idx for idx, val in enumerate(particle_lattice) if val == 0], dtype=np.int64)

    randgauss = np.random.randn(N, 3)
    randspins = randgauss / np.linalg.norm(randgauss, axis = 1, keepdims = True) # See https://mathworld.wolfram.com/SpherePointPicking.html (16).

    # Give particles spins, which are saved as a separate spin_lattice.
    spin_lattice = np.zeros((N, 3))
    k = 0
    for idx_enum in range(N):
        if particle_lattice[idx_enum] == 1:
            spin_lattice[idx_enum, 0] = randspins[k, 0]
            spin_lattice[idx_enum, 1] = randspins[k, 1]
            spin_lattice[idx_enum, 2] = randspins[k, 2]
            k += 1

    E = 0.0
    for idx_enum in range(filled_coords.shape[0]):
        i = filled_coords[idx_enum]
        sx_init = spin_lattice[i, 0]
        sy_init = spin_lattice[i, 1]
        sz_init = spin_lattice[i, 2]
        nn_i = lattice_nn[i]

        for k in range(n_neighbours):
            i_neighbor = nn_i[k]
            V = V_list[k]
            J = J_list[k]
            E += ((V * <double>particle_lattice[i_neighbor])
             + (J * (sx_init * spin_lattice[i_neighbor, 0] + sy_init * spin_lattice[i_neighbor, 1] + sz_init * spin_lattice[i_neighbor, 2])))
    
    E /= 2
    E_array = []
    E_arrays = []

    schedule = np.round(schedule, 3)
    print("Simulation starting...")

    for T in schedule:
        # Sample random values for later use; fastest to sample in one go rather than every time we need a new random value.
        randints_filled = np.random.randint(0, filled_coords.shape[0], 2 * n_steps).astype(np.int64) # Half for the particle move, half for the spin move.
        randints_empty = np.random.randint(0, empty_coords.shape[0], n_steps).astype(np.int64)
        randfloats = np.random.random(2 * n_steps) # Half for particle move, half for spin move.
        randgauss = np.random.randn(n_steps, 3)
        randspins = randgauss / np.linalg.norm(randgauss, axis = 1, keepdims = True) # See https://mathworld.wolfram.com/SpherePointPicking.html (16).

        data_collection_counter = 0

        print(f"Starting simulation at T = {T}.")

        for step in range(0, n_steps):
            deltaE = 0.0

            ##############################################
            # MOVE 1: MOVE ONE PARTICLE TO AN EMPTY SITE #
            ##############################################

            # Choose a random (particle, empty site) pair from the lattice.
            i_idx = randints_filled[step]
            j_idx = randints_empty[step]

            # Retrieve the site coordinates, the particle's spin, and the site nearest neighbours.
            i = filled_coords[i_idx]
            j = empty_coords[j_idx]

            sx_init = spin_lattice[i, 0]
            sy_init = spin_lattice[i, 1]
            sz_init = spin_lattice[i, 2]

            nn_i = lattice_nn[i]
            nn_j = lattice_nn[j]

            # Calculate the contribution of i and j to the total energy before swapping; note site j is empty
            # and so has 0 contribution.           
            k = 0
            for k in range(n_neighbours):
                nn = nn_i[k]
                V = V_list[k]
                J = J_list[k]

                deltaE -= ((V * <double>particle_lattice[nn])
                 + (J * (sx_init * spin_lattice[nn, 0] + sy_init * spin_lattice[nn, 1] + sz_init * spin_lattice[nn, 2])))

            # Move the particle to empty site j, including swapping the spin values.
            particle_lattice[i] = 0
            particle_lattice[j] = 1
            
            tmp_x = spin_lattice[i, 0]; tmp_y = spin_lattice[i, 1]; tmp_z = spin_lattice[i, 2]
            spin_lattice[i, 0] = spin_lattice[j, 0]; spin_lattice[i, 1] = spin_lattice[j, 1]; spin_lattice[i, 2] = spin_lattice[j, 2]
            spin_lattice[j, 0] = tmp_x; spin_lattice[j, 1] = tmp_y; spin_lattice[j, 2] = tmp_z

            # Calculate the change in the contribution after moving, and hence get the deltaE of the move.
            k = 0
            for k in range(n_neighbours):
                nn = nn_j[k]
                V = V_list[k]
                J = J_list[k]

                deltaE += ((V * <double>particle_lattice[nn])
                 + (J * (sx_init * spin_lattice[nn, 0] + sy_init * spin_lattice[nn, 1] + sz_init * spin_lattice[nn, 2])))

            if deltaE < 0 or randfloats[step] < exp(-deltaE/T):
                # If the move is accepted, swap coords between lists of filled and empty site coordinates.
                filled_coords[i_idx], empty_coords[j_idx] = empty_coords[j_idx], filled_coords[i_idx] 
                E += deltaE
            else:
                # If the move is rejected, move the particle back.
                particle_lattice[i] = 1
                particle_lattice[j] = 0
                tmp_x = spin_lattice[i, 0]; tmp_y = spin_lattice[i, 1]; tmp_z = spin_lattice[i, 2]
                spin_lattice[i, 0] = spin_lattice[j, 0]; spin_lattice[i, 1] = spin_lattice[j, 1]; spin_lattice[i, 2] = spin_lattice[j, 2]
                spin_lattice[j, 0] = tmp_x; spin_lattice[j, 1] = tmp_y; spin_lattice[j, 2] = tmp_z

            deltaE = 0.0

            ###################################################
            # MOVE 2: RANDOMIZE THE SPIN OF A SINGLE PARTICLE #
            ###################################################

            # Choose a random particle on the lattice.
            i = filled_coords[randints_filled[step + n_steps]]

            # Get particle i's spin and nearest neighbours.
            sx_init = spin_lattice[i, 0]
            sy_init = spin_lattice[i, 1]
            sz_init = spin_lattice[i, 2]

            nn_i = lattice_nn[i]

            # Choose a new candidate random spin.
            sx_cand = randspins[step, 0]
            sy_cand = randspins[step, 1]
            sz_cand = randspins[step, 2]

            spin_lattice[i, 0] = sx_cand
            spin_lattice[i, 1] = sy_cand
            spin_lattice[i, 2] = sz_cand

            # Calculate the change in spin.
            sdx = sx_cand - sx_init
            sdy = sy_cand - sy_init
            sdz = sz_cand - sz_init

            # Calculate deltaE.
            k = 0
            for k in range(n_neighbours):
                nn = nn_i[k]
                J = J_list[k]

                deltaE += (J * (sdx * spin_lattice[nn, 0] + sdy * spin_lattice[nn, 1] + sdz * spin_lattice[nn, 2]))

            if deltaE < 0 or randfloats[step + n_steps] < exp(-deltaE/T):
                # If the spin change is accepted, change the energy.
                E += deltaE
            else:
                # If the spin change is rejected, undo the spin change.
                spin_lattice[i, 0] = sx_init
                spin_lattice[i, 1] = sy_init
                spin_lattice[i, 2] = sz_init

            ################
            # COLLECT DATA #
            ################

            if data_collection_counter == data_collection_step:

                E_array.append(E)
                
                """ 
                # Uncomment this if you want to check that the cumulative energy sum is correct.
                e = 0
                for i in filled_coords:

                    i_spin_initial = spin_lattice[i].copy()
                    nn_i = lattice_nn[i]

                    k = 0
                    for k in range(n_neighbours):

                        i = nn_i[k]
                        V = V_list[k]
                        J = J_list[k]
                        
                        e += (V * particle_lattice[i]) + (J * np.dot(i_spin_initial, spin_lattice[i]))
                
                e /= 2
                print("True E:", e, "Cumulative E:", E, "dE:", deltaE) 
                """

                data_collection_counter = 0

            # Begin collecting data when equilibrium is reached.

            elif step > thermalisation_step:  
                
                data_collection_counter += 1

        E_arrays.append(E_array)

        E_array = []

    print("Simulation complete.")

    # Save the output data.
    data_names = ["particle_lattice", "spin_lattice", "E_array", "n_sweeps", "data_collection_interval", "schedule"]
    dat_utils.save_data([particle_lattice, spin_lattice, E_arrays, n_sweeps, data_collection_interval, schedule], data_names)