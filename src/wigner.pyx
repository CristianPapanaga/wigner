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
    for idx_enum, val_enum in enumerate(particle_lattice):
        if val_enum == 1:
            spin_lattice[idx_enum] = randspins[k]
            k += 1

    # Get energy. We only need to loop over occupied sites; unoccupied sites contribute 0 to the energy.
    E = 0.0
    for i in filled_coords:
        i_spin_initial = spin_lattice[i]
        nn_i = lattice_nn[i]

        k = 0
        for k in range(n_neighbours):
            i_neighbor = nn_i[k]
            V = V_list[k]
            J = J_list[k]

            E += (V * <double>particle_lattice[i_neighbor]) + (J * np.dot(i_spin_initial, spin_lattice[i_neighbor]))
    
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

            # Choose a random (particle, empty site) pair from the lattice
            # and retrieve relevant information about those sites.

            i_idx = randints_filled[step]
            j_idx = randints_empty[step]

            i = filled_coords[i_idx]
            j = empty_coords[j_idx]

            i_spin_initial = spin_lattice[i].copy()
            nn_i = lattice_nn[i]
            nn_j = lattice_nn[j]

            # Calculate the contribution of i and j to the total energy before swapping; note site j is empty
            # and so has 0 contribution.           
            k = 0
            for k in range(n_neighbours):
                nn = nn_i[k]
                V = V_list[k]
                J = J_list[k]

                deltaE -= ((V * <double>particle_lattice[nn]) + (J * np.dot(i_spin_initial, spin_lattice[nn])))

            # Move the particle to empty site j.
            particle_lattice[i] = 0
            particle_lattice[j] = 1
            spin_lattice[[i, j]] = spin_lattice[[j, i]]

            # Calculate the change in the contribution after moving, and hence get the deltaE of the move.
            k = 0
            for k in range(n_neighbours):
                nn = nn_j[k]
                V = V_list[k]
                J = J_list[k]

                deltaE += ((V * <double>particle_lattice[nn]) + (J * np.dot(i_spin_initial, spin_lattice[nn])))

            if deltaE < 0 or randfloats[step] < exp(-deltaE/T):
                # If the move is accepted, swap coords between lists of filled and empty site coordinates
                # and make sure to swap spin_lattice sites as well.
                filled_coords[i_idx], empty_coords[j_idx] = empty_coords[j_idx], filled_coords[i_idx] 
                E += deltaE
            else:
                # Move the particle back if the move was not accepted.
                particle_lattice[i] = 1
                particle_lattice[j] = 0
                spin_lattice[[i, j]] = spin_lattice[[j, i]]

            deltaE = 0.0

            ###################################################
            # MOVE 2: RANDOMIZE THE SPIN OF A SINGLE PARTICLE #
            ###################################################

            # Choose a random particle on the lattice.
            i = filled_coords[randints_filled[step + n_steps]]
            i_spin_initial = spin_lattice[i].copy()
            nn_i = lattice_nn[i]

            # Choose a new candidate random spin.
            i_spin_candidate = randspins[step]
            spin_lattice[i] = i_spin_candidate # This needs to be updated, as for small lattices the spin may self-interact.
            i_spin_diff = i_spin_candidate - i_spin_initial

            # Calculate deltaE.
            k = 0
            for k in range(n_neighbours):
                nn = nn_i[k]
                J = J_list[k]

                deltaE += (J * np.dot(i_spin_diff, spin_lattice[nn]))

            if deltaE < 0 or randfloats[step + n_steps] < exp(-deltaE/T):
                E += deltaE
            else:
                spin_lattice[i] = i_spin_initial

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